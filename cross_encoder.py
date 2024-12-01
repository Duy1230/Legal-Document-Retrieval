from sentence_transformers import CrossEncoder
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers import LoggingHandler
import logging
import pandas as pd
import torch


class LoggingCallback:
    def __init__(self):
        self.current_step = 0

    def __call__(self, score: float, epoch: int, steps: int):
        self.current_step += 1
        print(f'Step: {self.current_step}, Epoch: {epoch}, Loss: {score:.4f}')


class CrossEncoderWrapper:
    def __init__(
        self,
        model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
        max_length: int = 256,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Explicitly set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with explicit device
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=self.device
        )

    def train(self, train_dataset: Dataset, val_dataset: Dataset = None,
              num_epochs: int = 1, warmup_ratio: float = 0.02):
        """Train the cross-encoder model"""

       # Setup logging
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])

        # Prepare training examples
        train_samples = [
            InputExample(texts=[sample['question'], sample['document']],
                         label=sample['label'])
            for sample in train_dataset
        ]

        # Create data loader
        train_dataloader = DataLoader(
            train_samples,
            shuffle=True,
            batch_size=self.batch_size
        )

        # Create evaluator for validation
        if val_dataset is not None:
            # Prepare validation data in the required format
            val_samples = [
                InputExample(texts=[sample['question'], sample['document']],
                             label=sample['label'])
                for sample in val_dataset
            ]

            evaluator = CEBinaryAccuracyEvaluator.from_input_examples(
                val_samples, name="evaluation")
        else:
            evaluator = None
        # Create callback
        callback = LoggingCallback()

        # Train the model
        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=2000,  # Evaluate every 2000 steps
            warmup_steps=int(len(train_dataset) * warmup_ratio),
            show_progress_bar=True,
            output_path='checkpoints',  # Save checkpoints
            # save_best_model=True,
            callback=callback  # Add the callback
        )

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model on a dataset"""
        pairs = [
            [sample['question'], sample['document']]
            for sample in dataset
        ]
        labels = [sample['label'] for sample in dataset]

        # Get predictions
        scores = self.model.predict(pairs)
        predictions = (scores > 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'auc_roc': roc_auc_score(labels, scores)
        }

        return metrics

    def predict(self, questions: List[str], documents: List[str], show_progress_bar=True) -> np.ndarray:
        """Get relevance scores for question-document pairs"""
        try:
            # Ensure model is in eval mode
            self.model.model.eval()
            
            # Create pairs
            pairs = [[q, d] for q, d in zip(questions, documents)]
            
            # Make prediction with smaller batch size
            return self.model.predict(
                pairs,
                batch_size=8,  # Smaller batch size
                show_progress_bar=show_progress_bar
            )
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

    def save_model(self, path: str):
        self.model.save(path)

    def load_model(self, path: str):
        try:
            # First move model to CPU
            self.model.model = self.model.model.cpu()
            torch.cuda.empty_cache()  # Clear CUDA cache
            
            # Load the model with same configuration
            self.model = CrossEncoder(
                path,
                max_length=self.max_length,
                device=self.device
            )
            
            # Ensure model is in eval mode
            self.model.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise


class CrossEncoderDataset(Dataset):
    def __init__(self, questions: List[str], documents: List[str], labels: List[int]):
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


# This is to load saved dataset (the result of bi-encoder)
df = pd.read_csv("../input/cross-encoder-dataset-segmented/train_data.csv")
df = df.drop_duplicates()
df = df.sample(frac=1)
df.drop(columns=['cid', 'Unnamed: 0'], inplace=True)

train_data = df[:230000]
val_data = df[230000:240000]
test_data = df[240000:]

# Create train, val and test dataset
train_dataset = CrossEncoderDataset(
    questions=train_data['question'].tolist(),
    documents=train_data['document'].tolist(),
    labels=train_data['label'].tolist(),
)

val_dataset = CrossEncoderDataset(
    questions=val_data['question'].tolist(),
    documents=val_data['document'].tolist(),
    labels=val_data['label'].tolist(),
)

test_dataset = CrossEncoderDataset(
    questions=test_data['question'].tolist(),
    documents=test_data['document'].tolist(),
    labels=test_data['label'].tolist(),
)

# Initialize
model = CrossEncoderWrapper(
    model_name="bkai-foundation-models/vietnamese-bi-encoder",
    max_length=256,
    batch_size=16
)

# Train
model.train(train_dataset, val_dataset, num_epochs=1)


# Load model
model = CrossEncoderWrapper(
    model_name="bkai-foundation-models/vietnamese-bi-encoder",
    max_length=256,
    batch_size=16
)

# Load trained model and predict on 2000 sample of test dataset
model.load_model('/kaggle/input/cross_encoder_new/pytorch/default/1')

scores = model.predict(test_dataset[:2000]['question'].tolist(), test_dataset[:2000]['document'].tolist())
