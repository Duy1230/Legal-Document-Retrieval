from dual_encoder import BiEncoder, BiEncoderConfig
from cross_encoder import CrossEncoderWrapper
import faiss
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Union
from tqdm import tqdm

class RetrievalPipeline:
    def __init__(self, 
                 bi_encoder_path: str,
                 cross_encoder_path: str,
                 faiss_index_path: str,
                 #embeddings_path: str,
                 corpus_df_path: str,
                 top_k: int = 50,
                 rerank_k: int = 10):
        """
        Initialize retrieval pipeline
        Args:
            bi_encoder_path: Path to bi encoder checkpoint
            cross_encoder_path: Path to cross encoder checkpoint
            faiss_index_path: Path to saved FAISS index
            embeddings_path: Path to saved document embeddings
            corpus_df_path: Path to corpus CSV file
            top_k: Number of candidates to retrieve from FAISS
            rerank_k: Number of final results after reranking
        """
        # Load models
        self.Bi_encoder = BiEncoder(BiEncoderConfig())
        self.cross_encoder = CrossEncoderWrapper()
        self.load_models(bi_encoder_path, cross_encoder_path)
        
        # Load FAISS index and embeddings
        self.index = faiss.read_index(faiss_index_path)
        #self.document_embeddings = np.load(embeddings_path)
        
        # Load corpus for retrieving text
        self.corpus_df = pd.read_csv(corpus_df_path)
        
        # Configuration
        self.top_k = top_k
        self.rerank_k = rerank_k
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bi_encoder.to(self.device)
        #self.cross_encoder.model.to(torch.device("cuda"))

    def load_models(self, bi_encoder_path: str, cross_encoder_path: str):
        bi_encoder_checkpoint = torch.load(bi_encoder_path)
        self.bi_encoder.document_encoder.load_state_dict(
        bi_encoder_checkpoint['document_encoder'])
        self.bi_encoder.question_encoder.load_state_dict(
        bi_encoder_checkpoint['question_encoder'])
        self.bi_encoder.config = bi_encoder_checkpoint['config']
        
        self.cross_encoder.load_model(cross_encoder_path)


    def retrieve(self, query: str, rerank: bool = True) -> List[Dict[str, Union[str, float, int]]]:
        """
        Retrieve and rerank documents for a query
        Args:
            query: Question string
        Returns:
            List of dicts containing retrieved documents with scores and metadata
        """
        # Clear cache
        torch.cuda.empty_cache()

        # Stage 1: Bi-encoder retrieval
        query_embedding = self.bi_encoder.encode_question([query])
        
        query_embedding = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, doc_indices = self.index.search(query_embedding, self.top_k)
        
        # Get candidate documents
        candidates = []
        for score, doc_idx in zip(scores[0], doc_indices[0]):
            candidates.append({
                'text': self.corpus_df.iloc[doc_idx]['text'],
                'cid': self.corpus_df.iloc[doc_idx]['cid'],
                'bi_encoder_score': float(score)
            })
        if rerank:
            # Stage 2: Cross-encoder reranking
            candidate_texts = [c['text'] for c in candidates]
            #self.bi_encoder.to(torch.device("cuda"))
            batch_size = 16  # Adjust this based on your GPU memory capacity
            cross_encoder_scores = []

            for i in range(0, len(candidate_texts), batch_size):
                batch_texts = candidate_texts[i:i + batch_size]
                batch_scores = self.cross_encoder.predict(
                        [query] * len(batch_texts),
                        batch_texts,
                        show_progress_bar=False
                    )
                cross_encoder_scores.extend(batch_scores)
            #self.cross_encoder.to(torch.device("cpu"))
            #Add cross-encoder scores
            for idx, score in enumerate(cross_encoder_scores):
                candidates[idx]['cross_encoder_score'] = float(score)
                
            candidates.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            results = candidates[:self.rerank_k]
        
            return results

        else:
            candidates.sort(key=lambda x: x['bi_encoder_score'], reverse=True)
            results = candidates[:self.rerank_k]

            return results

    def batch_retrieve(self, 
                      queries: List[str],
                      batch_size: int = 32) -> List[List[Dict[str, Union[str, float, int]]]]:
        """
        Batch retrieval for multiple queries
        """
        all_results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = [self.retrieve(q) for q in batch]
            all_results.extend(batch_results)
        return all_results
    
    
def evaluate_retrieval(pipeline, test_df, top_k=10, re_ranking=True):
    """
    Evaluate the retrieval pipeline on a test set using MAP, MRR, NDCG, and Recall@k.
    
    Args:
        pipeline: The retrieval pipeline instance.
        test_df: DataFrame containing 'question' and 'cid' columns.
        top_k: Number of top results to consider for evaluation.
        
    Returns:
        A dictionary with MAP, MRR, NDCG, and Recall@k scores.
    """
    def parse_cid_string(cid_str):
        """Parse a space-separated string of CIDs into a set of integers."""
        return set(map(int, cid_str.strip('[]').split()))

    true_cids = test_df['cid'].apply(parse_cid_string)
    questions = test_df['question'].tolist()
    
    average_precisions = []
    reciprocal_ranks = []
    ndcg_scores = []
    recall_at_k = []
    
    for question, true_cid_set in tqdm(zip(questions, true_cids), total=len(questions), desc="Evaluating"):
        # Retrieve documents
        results = pipeline.retrieve(question, re_ranking)
        
        # Get retrieved cids
        retrieved_cids = [result['cid'] for result in results[:top_k]]
        
        # Calculate Average Precision
        num_relevant = 0
        precision_sum = 0.0
        for i, cid in enumerate(retrieved_cids):
            if cid in true_cid_set:
                num_relevant += 1
                precision_sum += num_relevant / (i + 1)
        average_precision = precision_sum / len(true_cid_set) if true_cid_set else 0
        average_precisions.append(average_precision)
        
        # Calculate Reciprocal Rank
        reciprocal_rank = 0.0
        for i, cid in enumerate(retrieved_cids):
            if cid in true_cid_set:
                reciprocal_rank = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(reciprocal_rank)
        
        # Calculate NDCG
        dcg = 0.0
        idcg = sum(1.0 / (i + 1) for i in range(min(len(true_cid_set), top_k)))
        for i, cid in enumerate(retrieved_cids):
            if cid in true_cid_set:
                dcg += 1.0 / (i + 1)
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        # Calculate Recall@k
        relevant_retrieved = len(set(retrieved_cids) & true_cid_set)
        recall = relevant_retrieved / len(true_cid_set) if true_cid_set else 0
        recall_at_k.append(recall)
    
    # Calculate average metrics
    map_score = sum(average_precisions) / len(average_precisions)
    mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    avg_recall_at_k = sum(recall_at_k) / len(recall_at_k)
    
    return {
        'MAP': map_score,
        'MRR': mrr_score,
        'NDCG': avg_ndcg,
        'Recall@k': avg_recall_at_k
    }