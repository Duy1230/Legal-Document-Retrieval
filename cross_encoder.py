from torch.utils.data import Dataset
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import get_linear_schedule_with_warmup


class CrossEncoderConfig:
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 256,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay


class CrossEncoder(nn.Module):
    def __init__(self, config: CrossEncoderConfig):
        super().__init__()
        self.config = config

        # Load base model and tokenizer
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Simple scoring head
        hidden_size = self.encoder.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, questions: List[str], documents: List[str]) -> torch.Tensor:
        # Prepare input pairs
        pairs = [
            (q, d) for q, d in zip(questions, documents)
        ]

        # Tokenize with special tokens and attention masks
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Get contextualized embeddings
        outputs = self.encoder(**inputs)

        # Use [CLS] token embedding for scoring
        cls_embedding = outputs.last_hidden_state[:, 0]

        # Generate similarity score
        scores = self.score_head(cls_embedding)

        return scores

    @property
    def device(self):
        return next(self.parameters()).device


class CrossEncoderTrainer:
    def __init__(self, config: CrossEncoderConfig):
        self.config = config
        self.model = CrossEncoder(config)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_dataset: Dataset, val_dataset: Dataset = None,
              patience: int = 3, min_delta: float = 0.001, eval_steps: int = 1000):
        """
        Train the model with early stopping and regular evaluation

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            patience: Number of evaluations to wait for improvement before stopping
            min_delta: Minimum change in validation score to qualify as an improvement
            eval_steps: Number of steps between evaluations
        """
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Initialize optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        num_training_steps = len(train_dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        criterion = nn.BCEWithLogitsLoss()

        # Early stopping variables
        best_val_score = 0
        best_model_state = None
        patience_counter = 0
        global_step = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            epoch_steps = 0

            progress_bar = tqdm(train_dataloader, desc=f'Epoch {
                                epoch + 1}/{self.config.num_epochs}')

            for batch in progress_bar:
                questions = batch['question']
                documents = batch['document']
                labels = batch['label'].float().to(self.device)

                scores = self.model(questions, documents)
                loss = criterion(scores.squeeze(), labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': total_loss / epoch_steps,
                    'step': global_step
                })

                # Evaluate at regular intervals
                if global_step % eval_steps == 0 and val_dataset:
                    avg_loss = total_loss / epoch_steps
                    print(f'\nStep {global_step}, Average Loss: {
                          avg_loss:.4f}')

                    val_metrics = self.evaluate(val_dataset)
                    print(f"Validation metrics: {val_metrics}")
                    current_val_score = val_metrics['auc_roc']

                    # Check if the model has improved
                    if current_val_score > best_val_score + min_delta:
                        print(f"Validation score improved from {
                              best_val_score:.4f} to {current_val_score:.4f}")
                        best_val_score = current_val_score
                        best_model_state = self.model.state_dict().copy()
                        patience_counter = 0
                        self.save_model('best_cross_encoder.pt')
                    else:
                        patience_counter += 1
                        print(f"Validation score did not improve. Patience: {
                              patience_counter}/{patience}")

                    # Early stopping check
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at step {
                              global_step}")
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        return

                    # Switch back to training mode
                    self.model.train()

            # End of epoch logging
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

        # If we completed all epochs, ensure we're using the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                questions = batch['question']
                documents = batch['document']
                labels = batch['label']

                scores = self.model(questions, documents)

                all_scores.extend(scores.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # Calculate metrics
        predictions = (all_scores > 0).astype(int)
        metrics = {
            'accuracy': accuracy_score(all_labels, predictions),
            'precision': precision_score(all_labels, predictions),
            'recall': recall_score(all_labels, predictions),
            'f1': f1_score(all_labels, predictions),
            'auc_roc': roc_auc_score(all_labels, all_scores)
        }

        return metrics

    def predict(self, questions: List[str], documents: List[str]) -> np.ndarray:
        """
        Get relevance scores for question-document pairs
        """
        self.model.eval()

        # Create simple dataset for prediction
        pairs = list(zip(questions, documents))
        batch_size = self.config.batch_size
        scores = []

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_questions = [p[0] for p in batch_pairs]
                batch_documents = [p[1] for p in batch_pairs]

                batch_scores = self.model(batch_questions, batch_documents)
                scores.extend(batch_scores.squeeze().cpu().numpy())

        return np.array(scores)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']


class CrossEncoderDataset(Dataset):
    def __init__(self, questions: List[str], documents: List[str], labels: List[int]):
        """
        Dataset for cross-encoder training and evaluation

        Args:
            questions: List of questions
            documents: List of documents
            labels: List of labels (1 for relevant, 0 for not relevant)
        """
        self.questions = questions
        self.documents = documents
        self.labels = labels

        assert len(questions) == len(documents) == len(
            labels), "All inputs must have the same length"

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            'question': self.questions[idx],
            'document': self.documents[idx],
            'label': self.labels[idx]
        }
