from dual_encoder import BiEncoder
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List


def build_faiss_index(
    corpus_path: str,
    model: BiEncoder,
    batch_size: int = 32,
    index_path: str = "document_index.faiss",
    embeddings_path: str = "document_embeddings.npy",
    nlist: int = 1000,  # Number of clusters/cells
    nprobe: int = 100,  # Number of cells to visit during search
) -> faiss.IndexIVFFlat:
    """
    Build optimized FAISS index from corpus documents

    Args:
        corpus_path: Path to corpus CSV file
        model: Trained BiEncoder model
        batch_size: Batch size for encoding
        index_path: Where to save the FAISS index
        nlist: Number of Voronoi cells (clusters)
        nprobe: Number of nearest cells to search
    """
    print("Reading corpus...")
    df = pd.read_csv(corpus_path)
    documents = df['text'].tolist()

    # Prepare model
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize lists to store batched embeddings
    all_embeddings = []

    # Process in batches to handle memory efficiently
    print("Encoding documents...")

    progress_bar = tqdm(total=int(len(documents)/batch_size),
                        desc="Encoding documents",
                        ncols=80,
                        position=0,  # Force position to 0
                        leave=True)  # Keep final result visible

    with torch.no_grad():
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = model.encode_document(
                batch, disable_progress_bar=True)
            all_embeddings.append(embeddings.cpu())
            progress_bar.update(1)

    # Concatenate all embeddings
    document_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    dimension = document_embeddings.shape[1]

    # Normalize embeddings
    faiss.normalize_L2(document_embeddings)

    # Create GPU resource
    res = faiss.StandardGpuResources()

    # Configure index
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(
        quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    # Transfer to GPU for training and adding vectors
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    # Train index
    print("Training index...")
    gpu_index.train(document_embeddings)

    # Add vectors to index
    print("Adding vectors to index...")
    gpu_index.add(document_embeddings)

    # Set number of cells to probe during search
    gpu_index.nprobe = nprobe

    # Transfer back to CPU for saving
    index = faiss.index_gpu_to_cpu(gpu_index)

    # Save index
    print("Saving index...")
    faiss.write_index(index, index_path)

    #  Save Embeddings
    print("Saving embeddings...")
    np.save(embeddings_path, document_embeddings)

    print(f"Index built with {index.ntotal} vectors of dimension {dimension}")
    print(f"Number of clusters: {nlist}, nprobe: {nprobe}")
    return index, document_embeddings


def prepare_cross_encoder_data(
    train_df: pd.DataFrame,
    bi_encoder_model: BiEncoder,
    corpus_df: pd.DataFrame,
    num_hard_negatives: int = 3,
    num_random_negatives: int = 1,
    batch_size: int = 32,
    embeddings_path: str = "document_embeddings.npy"
) -> pd.DataFrame:
    """
    Prepare training data for cross-encoder with multiple types of negatives using pre-computed document embeddings

    Args:
        train_df: Training dataframe with questions and ground truths
        bi_encoder_model: Trained bi-encoder model for finding hard negatives
        corpus_df: Full corpus dataframe
        num_hard_negatives: Number of hard negatives per question
        num_random_negatives: Number of random negatives per question
        batch_size: Batch size for bi-encoder inference
        embeddings_path: Path to saved document embeddings
    """
    training_pairs = []

    # Create corpus lookup
    corpus_lookup = dict(zip(corpus_df['cid'], corpus_df['text']))

    def parse_context(context_str: str) -> List[str]:
        """Parse context string with single-quote separated format"""
        if isinstance(context_str, str):
            # Remove outer brackets and split by single quotes
            cleaned = context_str.strip('[]')
            # Split by single quotes, filter out empty strings
            return [s for s in cleaned.split("'") if s.strip()]
        return context_str

    def parse_cids(cid_str: str) -> List[int]:
        """Parse space-separated CIDs"""
        if isinstance(cid_str, str):
            # Remove brackets and split by whitespace
            return [int(cid) for cid in cid_str.strip('[]').split()]
        return cid_str

    # Load pre-computed document embeddings
    print("Loading pre-computed document embeddings...")
    d_embeddings = torch.from_numpy(np.load(embeddings_path))

    # Process questions in batches with accurate progress tracking
    training_pairs = []
    total_questions = len(train_df)

    with tqdm(total=total_questions, desc="Processing questions") as pbar:
        for start_idx in range(0, len(train_df), batch_size):
            batch_df = train_df.iloc[start_idx:start_idx + batch_size]

            # Get question embeddings for the batch
            questions = batch_df['question'].tolist()
            q_embeddings = bi_encoder_model.encode_question(questions)

            # Compute similarities
            similarities = torch.matmul(q_embeddings, d_embeddings.t())

            # Process each question in the batch
            for batch_idx, (_, row) in enumerate(batch_df.iterrows()):
                question = row['question']

                # Parse context and CIDs
                correct_docs = parse_context(row['context'])
                correct_cids = parse_cids(row['cid'])

                # Add positive pairs
                for doc, cid in zip(correct_docs, correct_cids):
                    training_pairs.append({
                        'question': question,
                        'document': doc,
                        'label': 1,
                        'cid': int(cid)
                    })

                # Get hard negatives
                q_sim = similarities[batch_idx]
                _, candidate_indices = q_sim.topk(
                    num_hard_negatives + len(correct_cids))

                # Filter out correct documents
                hard_negative_indices = [
                    idx.item() for idx in candidate_indices
                    if corpus_df.iloc[int(idx.item())]['cid'] not in correct_cids
                ][:num_hard_negatives]

                # Add hard negative pairs
                for neg_idx in hard_negative_indices:
                    neg_doc = corpus_df.iloc[int(neg_idx)]
                    training_pairs.append({
                        'question': question,
                        'document': neg_doc['text'],
                        'label': 0,
                        'cid': neg_doc['cid']
                    })

                # Add random negatives
                random_negative_cids = np.random.choice(
                    [cid for cid in corpus_df['cid'] if cid not in correct_cids],
                    size=num_random_negatives,
                    replace=False
                )

                for neg_cid in random_negative_cids:
                    training_pairs.append({
                        'question': question,
                        'document': corpus_lookup[neg_cid],
                        'label': 0,
                        'cid': neg_cid
                    })

                # Update progress bar for each processed question
                pbar.update(1)
    return pd.DataFrame(training_pairs)
