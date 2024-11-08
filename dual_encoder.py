from sklearn.model_selection import train_test_split
import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple


class DualEncoderConfig:
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 256,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        temperature: float = 0.05,
        embedding_dim: int = 768  # Default XLM-RoBERTa hidden size
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.embedding_dim = embedding_dim


class DualEncoder(nn.Module):
    def __init__(self, config: DualEncoderConfig):
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


class DualEncoderTrainer:
    def get_config(self) -> DualEncoderConfig:
        """Get the trainer's configuration"""
        return self.config

    def get_model(self) -> DualEncoder:
        """Get the trainer's model"""
        return self.model

    def __init__(self, config: DualEncoderConfig):
        self.config = config
        self.model = DualEncoder(config.model_name, config.max_length)

        # Setup multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model.question_encoder = nn.DataParallel(
                self.model.question_encoder)
            self.model.document_encoder = nn.DataParallel(
                self.model.document_encoder)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Add metrics tracking
        self.best_mrr = 0.0
        self.best_recall = 0.0

    def prepare_batch(self, batch: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        # Tokenize questions
        questions_tokenized = self.model.tokenizer(
            batch['question'],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)

        # Tokenize contexts
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
        # Regular in-batch negative loss
        similarity = torch.matmul(q_embeddings, d_embeddings.t())

        if hard_negative_embeddings is not None:
            # Add hard negative similarities
            hard_similarity = torch.matmul(
                q_embeddings, hard_negative_embeddings.t())
            similarity = torch.cat([similarity, hard_similarity], dim=1)

        # Scale by temperature
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
                    self.model.question_encoder(
                        **processed_batch['question_data']),
                    processed_batch['question_data']['attention_mask']
                )
                d_embeddings = self.model.mean_pooling(
                    self.model.document_encoder(
                        **processed_batch['context_data']),
                    processed_batch['context_data']['attention_mask']
                )

                # Normalize
                q_embeddings = nn.functional.normalize(
                    q_embeddings, p=2, dim=1)
                d_embeddings = nn.functional.normalize(
                    d_embeddings, p=2, dim=1)

                all_q_embeddings.append(q_embeddings)
                all_d_embeddings.append(d_embeddings)

        # Concatenate all embeddings
        all_q_embeddings = torch.cat(all_q_embeddings, dim=0)
        all_d_embeddings = torch.cat(all_d_embeddings, dim=0)

        # Compute similarity matrix
        similarity = torch.matmul(all_q_embeddings, all_d_embeddings.t())

        # Calculate metrics
        k_values = [1, 5, 10, 50]
        metrics = {}

        for k in k_values:
            # Get top-k indices
            _, indices = similarity.topk(k, dim=1)

            # Calculate Recall@k
            correct = torch.arange(similarity.size(0)).unsqueeze(
                1).expand_as(indices).to(self.device)
            recall_at_k = (indices == correct).float().sum(dim=1).mean().item()
            metrics[f'recall@{k}'] = recall_at_k

            # Calculate MRR@k
            rank = (indices == correct).nonzero()[:, 1] + 1
            mrr = (1.0 / rank).mean().item()
            metrics[f'mrr@{k}'] = mrr

        return metrics

    def train(self, train_dataset: LegalDataset, val_dataset: LegalDataset = None):
        # Enable automatic mixed precision training
        scaler = torch.amp.GradScaler(device_type='cuda')

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Separate optimizers for each encoder
        question_optimizer = torch.optim.AdamW(
            self.model.question_encoder.parameters(),
            lr=self.config.learning_rate
        )
        document_optimizer = torch.optim.AdamW(
            self.model.document_encoder.parameters(),
            lr=self.config.learning_rate
        )

        # Gradient accumulation steps
        gradient_accumulation_steps = 4  # Simulate batch_size=32 with batch_size=8

        # Mine hard negatives every n epochs
        mine_every_n_epochs = 1

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {
                                epoch + 1}/{self.config.num_epochs}')

            # Mine hard negatives at the start of specified epochs
            hard_negatives = None
            if epoch > 0 and epoch % mine_every_n_epochs == 0:
                print("Mining hard negatives...")
                questions = train_dataset.questions
                contexts = train_dataset.contexts
                hard_negatives = self.mine_hard_negatives(questions, contexts)

            for batch_idx, batch in enumerate(progress_bar):
                # Zero gradients only at the start of accumulation steps
                if batch_idx % gradient_accumulation_steps == 0:
                    question_optimizer.zero_grad()
                    document_optimizer.zero_grad()

                # Prepare batch data
                processed_batch = self.prepare_batch(batch)

                # Use automatic mixed precision
                with torch.amp.autocast(device_type='cuda'):
                    # Get embeddings from separate encoders
                    q_embeddings = self.model.mean_pooling(
                        self.model.question_encoder(
                            **processed_batch['question_data']),
                        processed_batch['question_data']['attention_mask']
                    )
                    d_embeddings = self.model.mean_pooling(
                        self.model.document_encoder(
                            **processed_batch['context_data']),
                        processed_batch['context_data']['attention_mask']
                    )

                    # Process hard negatives if available
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
                        hard_negative_embeddings = nn.functional.normalize(
                            hard_negative_embeddings, p=2, dim=1)

                    # Normalize embeddings
                    q_embeddings = nn.functional.normalize(
                        q_embeddings, p=2, dim=1)
                    d_embeddings = nn.functional.normalize(
                        d_embeddings, p=2, dim=1)

                    # Compute loss with hard negatives
                    loss = self.compute_loss(
                        q_embeddings, d_embeddings, hard_negative_embeddings)
                    loss = loss / gradient_accumulation_steps  # Scale loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Update weights only after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Unscale gradients and step optimizers
                    question_optimizer.step()
                    question_optimizer.zero_grad()
                    document_optimizer.step()
                    document_optimizer.zero_grad()

                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)

                    scaler.step(question_optimizer)
                    scaler.step(document_optimizer)
                    scaler.update()

                total_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix(
                    {'loss': total_loss / (batch_idx + 1)})

                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            avg_loss = total_loss / len(train_dataloader)
            print(
                f'Epoch {epoch + 1}/{self.config.num_epochs}, Average Loss: {avg_loss:.4f}')

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
        self.model.question_encoder.load_state_dict(
            checkpoint['question_encoder'])
        self.model.document_encoder.load_state_dict(
            checkpoint['document_encoder'])

    def mine_hard_negatives(self, questions: List[str], documents: List[str],
                            batch_size: int = 512) -> List[str]:
        """Mine hard negatives using embeddings similarity"""
        self.model.eval()
        device = self.device  # Use the trainer's device

        # Process in smaller chunks to avoid OOM
        chunk_size = 10000  # Reduced chunk size to avoid OOM

        with torch.no_grad():
            all_q_embeddings = []
            for i in range(0, len(questions), chunk_size):
                q_chunk = questions[i:i + chunk_size]
                q_emb = self.model.encode_question(q_chunk, batch_size)
                all_q_embeddings.append(q_emb)
            q_embeddings = torch.cat(all_q_embeddings, dim=0)

            all_d_embeddings = []
            for i in range(0, len(documents), chunk_size):
                d_chunk = documents[i:i + chunk_size]
                d_emb = self.model.encode_document(d_chunk, batch_size)
                all_d_embeddings.append(d_emb)
            d_embeddings = torch.cat(all_d_embeddings, dim=0)

            # Free up memory
            torch.cuda.empty_cache()

            # Normalize embeddings
            q_embeddings = nn.functional.normalize(q_embeddings, p=2, dim=1)
            d_embeddings = nn.functional.normalize(d_embeddings, p=2, dim=1)

            # Compute similarity in chunks to avoid OOM
            hard_negative_indices = []
            chunk_size = 1000  # Reduced chunk size for similarity computation
            k = 3  # number of hard negatives per question

            for i in range(0, len(q_embeddings), chunk_size):
                q_chunk = q_embeddings[i:i + chunk_size]
                similarity = torch.matmul(q_chunk, d_embeddings.t())

                # Get top-k most similar but incorrect documents
                values, indices = similarity.topk(k + 1, dim=1)

                # Filter out positive pairs
                mask = torch.arange(i, min(
                    i + chunk_size, len(q_embeddings))).unsqueeze(1).expand_as(indices).to(device)
                chunk_negative_indices = indices[indices != mask].view(-1)
                hard_negative_indices.extend(
                    chunk_negative_indices.cpu().numpy())

                # Free memory
                del similarity
                torch.cuda.empty_cache()

            return [documents[idx] for idx in hard_negative_indices]

# Example usage


def prepare_training_data(train_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    questions = train_df['question'].tolist()
    contexts = train_df['context'].apply(lambda x: eval(
        x)[0] if isinstance(x, str) else x[0]).tolist()
    return questions, contexts


def train_dual_encoder(train_df: pd.DataFrame):
    # Initialize config
    config = DualEncoderConfig()

    # Prepare data
    questions, contexts = prepare_training_data(train_df)

    # Create dataset
    dataset = LegalDataset(
        questions=questions,
        contexts=contexts,
        tokenizer=XLMRobertaTokenizer.from_pretrained(config.model_name),
        max_length=config.max_length
    )

    # Initialize trainer
    trainer = DualEncoderTrainer(config)

    # Train model
    trainer.train(dataset)

    return trainer.model


def split_df(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=42)
