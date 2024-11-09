import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    XLMRobertaModel,
    XLMRobertaTokenizer,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass
from sklearn.metrics import average_precision_score, ndcg_score
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import py_vncorenlp


class CrossEncoderConfig:
    xlm_roberta_name: str = "xlm-roberta-base"
    phobert_name: str = "vinai/phobert-base"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    dropout_rate: float = 0.1
    ensemble_weights: List[float] = None  # If None, learned during training


class EnsembleCrossEncoder(nn.Module):
    def __init__(self, config: CrossEncoderConfig):
        super().__init__()
        self.config = config

        # XLM-RoBERTa encoder
        self.xlm_roberta = XLMRobertaModel.from_pretrained(
            config.xlm_roberta_name)
        self.xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(
            config.xlm_roberta_name)

        # PhoBERT encoder
        self.phobert = AutoModel.from_pretrained(config.phobert_name)
        self.pho_tokenizer = AutoTokenizer.from_pretrained(config.phobert_name)

        # Projection layers
        hidden_size = self.xlm_roberta.config.hidden_size
        self.xlm_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )

        self.pho_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

        # Ensemble weights (can be learned or fixed)
        if config.ensemble_weights is None:
            self.ensemble_weights = nn.Parameter(torch.ones(2))
        else:
            self.register_buffer('ensemble_weights',
                                 torch.tensor(config.ensemble_weights))

        # Initialize word segmenter
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir='/absolute/path/to/vncorenlp'
        )

    def segment_text(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Perform word segmentation for Vietnamese text
        Can handle both single string and list of strings
        """
        if isinstance(texts, str):
            # Single text
            try:
                segmented = self.rdrsegmenter.word_segment(texts)
                return ' '.join(segmented)
            except Exception as e:
                print(f"Segmentation error: {e}")
                return texts
        else:
            # List of texts
            segmented_texts = []
            for text in texts:
                try:
                    segmented = self.rdrsegmenter.word_segment(text)
                    segmented_texts.append(' '.join(segmented))
                except Exception as e:
                    print(f"Segmentation error for text: {text}\nError: {e}")
                    segmented_texts.append(text)
            return segmented_texts

    def forward(self,
                questions: Union[str, List[str]],
                documents: Union[str, List[str]],
                return_logits: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Add input validation
        if not questions or not documents:
            raise ValueError("Empty input received")
        
        # Convert single inputs to lists
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(documents, str):
            documents = [documents]
            
        if len(questions) != len(documents):
            raise ValueError(f"Mismatched input lengths: questions({len(questions)}) != documents({len(documents)})")

        # Segment text for PhoBERT
        segmented_questions = self.segment_text(questions)
        segmented_documents = self.segment_text(documents)
        
        # Prepare inputs for XLM-RoBERTa (using original text)
        xlm_inputs = self.xlm_tokenizer(
            questions,
            documents,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Prepare inputs for PhoBERT (using segmented text)
        pho_inputs = self.pho_tokenizer(
            segmented_questions,
            segmented_documents,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings from both models
        xlm_outputs = self.xlm_roberta(**xlm_inputs)
        pho_outputs = self.phobert(**pho_inputs)

        # Get CLS token embeddings
        xlm_cls = xlm_outputs.last_hidden_state[:, 0]
        pho_cls = pho_outputs.last_hidden_state[:, 0]

        # Project embeddings
        xlm_projected = self.xlm_projection(xlm_cls)
        pho_projected = self.pho_projection(pho_cls)

        # Normalize ensemble weights
        weights = torch.softmax(self.ensemble_weights, dim=0)

        # Weighted concatenation
        combined = torch.cat([
            weights[0] * xlm_projected,
            weights[1] * pho_projected
        ], dim=1)

        # Final classification
        logits = self.classifier(combined)

        if return_logits:
            return logits

        return {
            'logits': logits,
            'xlm_embedding': xlm_projected,
            'pho_embedding': pho_projected,
            'weights': weights
        }

    @property
    def device(self):
        return next(self.parameters()).device


class CrossEncoderDataset(Dataset):
    def __init__(self,
                 questions: List[str],
                 documents: List[str],
                 labels: List[int],
                 save_dir: str = '/absolute/path/to/vncorenlp'):
        self.questions = questions
        self.documents = documents
        self.labels = labels
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=save_dir
        )

    def segment_text(self, text: str) -> str:
        """
        Perform word segmentation
        """
        try:
            segmented = self.rdrsegmenter.word_segment(text)
            return ' '.join(segmented)
        except Exception as e:
            print(f"Segmentation error: {e}")
            return text

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        document = self.documents[idx]

        return {
            'question': question,
            'document': document,
            'label': self.labels[idx]
        }


class CrossEncoderTrainer:
    def __init__(self,
                 model: EnsembleCrossEncoder,
                 config: CrossEncoderConfig):
        self.model = model
        self.config = config
        
        # Add debugging info
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to device with error handling
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            print(f"Error moving model to device: {e}")
            print("Falling back to CPU")
            self.device = torch.device('cpu')
            self.model.to(self.device)

        # Track best metrics
        self.best_ndcg = 0.0
        self.best_map = 0.0

    def train(self,
              train_dataset: CrossEncoderDataset,
              val_dataset: CrossEncoderDataset = None,
              eval_steps: int = 1000):

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # Learning rate scheduler
        num_training_steps = len(train_dataloader) * self.config.num_epochs
        num_warmup_steps = num_training_steps // 10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(train_dataloader, desc=f'Epoch {
                                epoch + 1}/{self.config.num_epochs}')

            for batch in progress_bar:
                # Move batch to device
                questions = batch['question']
                documents = batch['document']
                labels = batch['label'].float().to(self.device)

                # Forward pass
                logits = self.model(questions, documents)
                loss = criterion(logits.squeeze(), labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})

                # Evaluate periodically
                if val_dataset and global_step % eval_steps == 0:
                    metrics = self.evaluate(val_dataset)
                    print("\nValidation metrics:")
                    for metric_name, value in metrics.items():
                        print(f"{metric_name}: {value:.4f}")

                    # Save best model
                    if metrics['ndcg@10'] > self.best_ndcg:
                        self.best_ndcg = metrics['ndcg@10']
                        self.save_model('best_cross_encoder.pt')

                    self.model.train()

            avg_loss = total_loss / len(train_dataloader)
            print(
                f'Epoch {epoch + 1}/{self.config.num_epochs}, Average Loss: {avg_loss:.4f}')

            # Full validation at end of epoch
            if val_dataset:
                metrics = self.evaluate(val_dataset)
                print("Validation metrics:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")

    def evaluate(self, dataset: CrossEncoderDataset) -> Dict[str, float]:
        """
        Evaluate the cross-encoder model using ranking metrics
        Returns NDCG@k and MAP@k
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                questions = batch['question']
                documents = batch['document']
                labels = batch['label']

                logits = self.model(questions, documents)
                scores = torch.sigmoid(logits).squeeze().cpu().numpy()

                all_scores.extend(scores)
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # Calculate metrics
        metrics = {}
        k_values = [1, 5, 10, 20]

        # Calculate NDCG@k
        for k in k_values:
            ndcg = ndcg_score(
                [all_labels],  # Expected shape: (n_queries, n_docs)
                [all_scores],  # Expected shape: (n_queries, n_docs)
                k=k
            )
            metrics[f'ndcg@{k}'] = ndcg

        # Calculate MAP (Mean Average Precision)
        map_score = average_precision_score(all_labels, all_scores)
        metrics['map'] = map_score

        return metrics

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']


def prepare_cross_encoder_data(
    train_df: pd.DataFrame,
    bi_encoder_model: DualEncoder,
    corpus_df: pd.DataFrame,
    num_hard_negatives: int = 3,
    num_random_negatives: int = 1,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Prepare training data for cross-encoder with multiple types of negatives

    Args:
        train_df: Training dataframe with questions and ground truths
        bi_encoder_model: Trained bi-encoder model for finding hard negatives
        corpus_df: Full corpus dataframe
        num_hard_negatives: Number of hard negatives per question
        num_random_negatives: Number of random negatives per question
        batch_size: Batch size for bi-encoder inference
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

    # Get all document embeddings once (this might take a while)
    print("Encoding all documents...")
    d_embeddings = bi_encoder_model.encode_document(corpus_df['text'].tolist())

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


def create_training_data(
    train_df: pd.DataFrame,
    bi_encoder_model: DualEncoder,
    corpus_df: pd.DataFrame,
    val_size: float = 0.1,
    vncorenlp_dir: str = '/absolute/path/to/vncorenlp'
) -> Tuple[CrossEncoderDataset, CrossEncoderDataset]:
    """
    Create training and validation datasets for cross-encoder
    """
    # Download VnCoreNLP if not already downloaded
    try:
        py_vncorenlp.download_model(save_dir=vncorenlp_dir)
    except Exception as e:
        print(f"Model already downloaded or download error: {e}")

    # Split into train and validation
    train_data, val_data = train_test_split(
        train_df, test_size=val_size, random_state=42)

    # Prepare training data with negatives
    train_pairs = prepare_cross_encoder_data(
        train_data,
        bi_encoder_model,
        corpus_df,
        num_hard_negatives=3,
        num_random_negatives=1
    )

    val_pairs = prepare_cross_encoder_data(
        val_data,
        bi_encoder_model,
        corpus_df,
        num_hard_negatives=2,
        num_random_negatives=1
    )

    # Create datasets
    train_dataset = CrossEncoderDataset(
        questions=train_pairs['question'].tolist(),
        documents=train_pairs['document'].tolist(),
        labels=train_pairs['label'].tolist(),
        save_dir=vncorenlp_dir
    )

    val_dataset = CrossEncoderDataset(
        questions=val_pairs['question'].tolist(),
        documents=val_pairs['document'].tolist(),
        labels=val_pairs['label'].tolist(),
        save_dir=vncorenlp_dir
    )

    return train_dataset, val_dataset

# Update the train_cross_encoder function


def train_cross_encoder(
    train_df: pd.DataFrame,
    bi_encoder_model: DualEncoder,
    corpus_df: pd.DataFrame
) -> EnsembleCrossEncoder:
    """
    Train the cross-encoder with improved data preparation
    """
    # Create training and validation datasets
    train_dataset, val_dataset = create_training_data(
        train_df,
        bi_encoder_model,
        corpus_df
    )

    # Initialize model and config
    config = CrossEncoderConfig()
    model = EnsembleCrossEncoder(config)

    # Initialize trainer
    trainer = CrossEncoderTrainer(model, config)

    # Train model
    trainer.train(train_dataset, val_dataset)

    return trainer.model
