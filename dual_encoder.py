import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
import faiss
import numpy as np
from sklearn.model_selection import train_test_split


class BiEncoderConfig:
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 256,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        temperature: float = 0.05,
        embedding_dim: int = 768  # Default XLM-RoBERTa hidden size,

    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.embedding_dim = embedding_dim


class BiEncoder(nn.Module):
    def __init__(self, config: BiEncoderConfig):
        super().__init__()
        self.config = config
        # Initialize two separate encoders
        self.question_encoder = XLMRobertaModel.from_pretrained(
            config.model_name)
        self.document_encoder = XLMRobertaModel.from_pretrained(
            config.model_name)

        # Shared tokenizer (we don't need separate tokenizers)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
        self.max_length = config.max_length

    def get_device(self):
        # Helper method to get the current device
        if isinstance(self.question_encoder, nn.DataParallel):
            return self.question_encoder.module.device
        return next(self.question_encoder.parameters()).device

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_question(self, questions: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        device = self.get_device()

        for i in range(0, len(questions), batch_size):
            batch_texts = questions[i:i + batch_size]

            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                model_output = self.question_encoder(**encoded_input)

            batch_embeddings = self.mean_pooling(
                model_output, encoded_input['attention_mask'])
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encode_document(self, documents: List[str], batch_size: int = 32, disable_progress_bar: bool = False) -> torch.Tensor:
        all_embeddings = []
        device = self.get_device()

        # Calculate total number of batches for progress bar
        num_batches = (len(documents) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(documents), batch_size), total=num_batches, desc="Encoding documents", disable=disable_progress_bar):
            batch_texts = documents[i:i + batch_size]

            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                model_output = self.document_encoder(**encoded_input)

            batch_embeddings = self.mean_pooling(
                model_output, encoded_input['attention_mask'])
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)


class LegalDataset(Dataset):
    def __init__(self, questions: List[str], contexts: List[str], tokenizer, max_length: int):
        self.questions = questions
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        
        return {
            'question': question,
            'context': context
        }
    

class BiEncoderTrainer:

    def get_config(self) -> BiEncoderConfig:
        """Get the trainer's configuration"""
        return self.config
        
    def get_model(self) -> BiEncoder:
        """Get the trainer's model"""
        return self.model
        
    def __init__(self, config: BiEncoderConfig):
        self.config = config
        self.model = BiEncoder(self.config)
        
        # Setup multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model.question_encoder = nn.DataParallel(self.model.question_encoder)
            self.model.document_encoder = nn.DataParallel(self.model.document_encoder)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # metrics tracking
        self.best_mrr = 0.0
        self.best_recall = 0.0

    def set_scores(self, scores: Tuple):
        self.best_mrr = scores[0]
        self.best_recall = scores[1]
        
    def prepare_batch(self, batch: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        # tokenize questions
        questions_tokenized = self.model.tokenizer(
            batch['question'],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        #Tokenize contexts
        contexts_tokenized = self.model.tokenizer(
            batch['context'],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        return {
            'question_data': questions_tokenized,
            'context_data': contexts_tokenized
        }
    
    def compute_loss(self, q_embeddings: torch.Tensor, d_embeddings: torch.Tensor,
                    hard_negative_embeddings: torch.Tensor = None) -> torch.Tensor:
        """Compute loss with both in-batch and hard negatives"""
        #regular in-batch negative loss
        similarity = torch.matmul(q_embeddings, d_embeddings.t())
        
        if hard_negative_embeddings is not None:
            # hard negative similarities
            hard_similarity = torch.matmul(q_embeddings, hard_negative_embeddings.t())
            similarity = torch.cat([similarity, hard_similarity], dim=1)
        
        # scale by temperature
        similarity = similarity / self.config.temperature
        
        # Create labels for diagonal (positive pairs)
        labels = torch.arange(q_embeddings.size(0)).to(self.device)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(similarity, labels)
        
        return loss
    
    def evaluate(self, val_dataset: LegalDataset) -> Dict[str, float]:
        """
        Evaluate the model on validation dataset
        Returns MRR@k and Recall@k metrics
        """
        self.model.eval()
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        all_q_embeddings = []
        all_d_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                processed_batch = self.prepare_batch(batch)
                
                # Get embeddings
                q_embeddings = self.model.mean_pooling(
                    self.model.question_encoder(**processed_batch['question_data']),
                    processed_batch['question_data']['attention_mask']
                )
                d_embeddings = self.model.mean_pooling(
                    self.model.document_encoder(**processed_batch['context_data']),
                    processed_batch['context_data']['attention_mask']
                )
                
                # Normalize
                q_embeddings = nn.functional.normalize(q_embeddings, p=2, dim=1)
                d_embeddings = nn.functional.normalize(d_embeddings, p=2, dim=1)
                
                all_q_embeddings.append(q_embeddings)
                all_d_embeddings.append(d_embeddings)
        
        # Concatenate all embeddings
        all_q_embeddings = torch.cat(all_q_embeddings, dim=0)
        all_d_embeddings = torch.cat(all_d_embeddings, dim=0)
        
        # Compute similarity matrix
        similarity = torch.matmul(all_q_embeddings, all_d_embeddings.t())
        
        # Calculate metrics
        k_values = [1, 5, 10, 50,100, 200, 500, 1000]
        metrics = {}
        
        for k in k_values:
            # Get top-k indices
            _, indices = similarity.topk(k, dim=1)
            
            # Calculate Recall@k
            correct = torch.arange(similarity.size(0)).unsqueeze(1).expand_as(indices).to(self.device)
            recall_at_k = (indices == correct).float().sum(dim=1).mean().item()
            metrics[f'recall@{k}'] = recall_at_k
            
            # Calculate MRR@k
            rank = (indices == correct).nonzero()[:, 1] + 1
            mrr = (1.0 / rank).mean().item()
            metrics[f'mrr@{k}'] = mrr
            
        return metrics

    def train(self, train_dataset: LegalDataset, val_dataset: LegalDataset = None):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        #separate optimizers for each encoder
        question_optimizer = torch.optim.AdamW(
            self.model.question_encoder.parameters(),
            lr=self.config.learning_rate
        )
        document_optimizer = torch.optim.AdamW(
            self.model.document_encoder.parameters(),
            lr=self.config.learning_rate
        )
        
        #mine hard negatives every n epochs
        mine_every_n_epochs = 1
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs}')
            
            # Mine hard negatives at the start of specified epochs
            hard_negatives = None
            if epoch > 0 and epoch % mine_every_n_epochs == 0:
                print("Mining hard negatives...")
                questions = train_dataset.questions
                contexts = train_dataset.contexts
                hard_negatives = self.mine_hard_negatives(questions, contexts)
            
            for batch_idx, batch in enumerate(progress_bar):
                question_optimizer.zero_grad()
                document_optimizer.zero_grad()
                
                #Prepare batch data
                processed_batch = self.prepare_batch(batch)
                
                #Get embeddings from separate encoders
                q_embeddings = self.model.mean_pooling(
                    self.model.question_encoder(**processed_batch['question_data']),
                    processed_batch['question_data']['attention_mask']
                )
                d_embeddings = self.model.mean_pooling(
                    self.model.document_encoder(**processed_batch['context_data']),
                    processed_batch['context_data']['attention_mask']
                )
                
                #Process hard negatives if available
                hard_negative_embeddings = None
                if hard_negatives is not None:
                    hard_negative_batch = self.model.tokenizer(
                        hard_negatives[batch_idx:batch_idx + len(batch)],
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    hard_negative_embeddings = self.model.mean_pooling(
                        self.model.document_encoder(**hard_negative_batch),
                        hard_negative_batch['attention_mask']
                    )
                    hard_negative_embeddings = nn.functional.normalize(hard_negative_embeddings, p=2, dim=1)
                
                #Normalize embeddings
                q_embeddings = nn.functional.normalize(q_embeddings, p=2, dim=1)
                d_embeddings = nn.functional.normalize(d_embeddings, p=2, dim=1)
                
                #Compute loss with hard negatives
                loss = self.compute_loss(q_embeddings, d_embeddings, hard_negative_embeddings)
                
                #Backward pass
                loss.backward()
                question_optimizer.step()
                document_optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
            
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1}/{self.config.num_epochs}, Average Loss: {avg_loss:.4f}')
            
            # Validation step
            if val_dataset is not None:
                metrics = self.evaluate(val_dataset)
                print("Validation metrics:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
                
                # Save best model
                if metrics['mrr@10'] > self.best_mrr:
                    self.best_mrr = metrics['mrr@10']
                    self.save_model('best_model.pt')
    
    def save_model(self, path: str):
        # Save both encoders and config
        torch.save({
            'question_encoder': self.model.question_encoder.state_dict(),
            'document_encoder': self.model.document_encoder.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        
        # Handle loading state dict with or without DataParallel
        question_state_dict = checkpoint['question_encoder']
        document_state_dict = checkpoint['document_encoder']
        
        # If using DataParallel now but the saved model wasn't
        if isinstance(self.model.question_encoder, nn.DataParallel):
            # Add 'module.' prefix to all keys if it doesn't exist
            question_state_dict = {'module.' + k: v for k, v in question_state_dict.items()}
            document_state_dict = {'module.' + k: v for k, v in document_state_dict.items()}
        
        self.model.question_encoder.load_state_dict(question_state_dict)
        self.model.document_encoder.load_state_dict(document_state_dict)

    def mine_hard_negatives(self, questions: List[str], documents: List[str], 
                       batch_size: int = 256) -> List[str]:  # Reduced batch size
        """Mine hard negatives using embeddings similarity"""
        self.model.eval()
        device = self.device
    
        # Significantly reduced chunk sizes to avoid OOM
        chunk_size = 2000  # Reduced from 5000
        similarity_chunk_size = 200  # Reduced from 500
    
        with torch.no_grad():
            # Process questions in smaller chunks
            all_q_embeddings = []
            for i in tqdm(range(0, len(questions), chunk_size), desc="Encoding questions"):
                q_chunk = questions[i:i + chunk_size]
                q_emb = self.model.encode_question(q_chunk, batch_size)
                all_q_embeddings.append(q_emb.cpu())  # Move to CPU after processing
                torch.cuda.empty_cache()
            q_embeddings = torch.cat(all_q_embeddings, dim=0)
        
            # Process documents in smaller chunks
            all_d_embeddings = []
            for i in tqdm(range(0, len(documents), chunk_size), desc="Encoding documents"):
                d_chunk = documents[i:i + chunk_size]
                d_emb = self.model.encode_document(d_chunk, batch_size)
                all_d_embeddings.append(d_emb.cpu())  # Move to CPU after processing
                torch.cuda.empty_cache()
            d_embeddings = torch.cat(all_d_embeddings, dim=0)
            
            # Normalize embeddings (on CPU to save GPU memory)
            q_embeddings = nn.functional.normalize(q_embeddings, p=2, dim=1)
            d_embeddings = nn.functional.normalize(d_embeddings, p=2, dim=1)
            
            # Compute similarity in smaller chunks
            hard_negative_indices = []
            k = 2  # Reduced number of hard negatives per question
            
            for i in tqdm(range(0, len(q_embeddings), similarity_chunk_size), desc="Mining negatives"):
                # Move only the current chunks to GPU
                q_chunk = q_embeddings[i:i + similarity_chunk_size].to(device)
                d_chunk = d_embeddings.to(device)
                
                similarity = torch.matmul(q_chunk, d_chunk.t())
                
                # Get top-k most similar but incorrect documents
                values, indices = similarity.topk(k + 1, dim=1)
                
                # Filter out positive pairs
                mask = torch.arange(i, min(i + similarity_chunk_size, len(q_embeddings))).unsqueeze(1).expand_as(indices).to(device)
                chunk_negative_indices = indices[indices != mask].view(-1)
                hard_negative_indices.extend(chunk_negative_indices.cpu().numpy())
                
                # Free memory
                del similarity, values, indices, q_chunk, d_chunk
                torch.cuda.empty_cache()
        
            return [documents[idx] for idx in hard_negative_indices]