import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data():
    try:
        df = pd.read_csv('data/processed/processed_data.csv')
        texts = df['processed_text'].fillna('').astype(str).tolist()
        labels = df['sentiment_encoded'].values
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        return train_texts, val_texts, train_labels, val_labels
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_accuracy = 0
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader)
        
        logger.info(f'Train loss: {avg_train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}')
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, tokenizer)
            logger.info(f'New best model saved with accuracy: {best_accuracy:.4f}')

def evaluate_model(model, data_loader):
    model.eval()
    val_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            val_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_val_loss, accuracy

def save_model(model, tokenizer, output_dir='models/distilbert_pt'):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f'Model and tokenizer saved to {output_dir}')

def save_metrics(metrics, filename='results/distilbert_metrics.json'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Metrics saved to {filename}')

if __name__ == "__main__":
    # Load and prepare data
    train_texts, val_texts, train_labels, val_labels = load_and_prepare_data()
    
    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(np.unique(train_labels))
    ).to(device)
    
    # Create datasets and data loaders
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=3)
    
    # Final evaluation
    logger.info('Final evaluation on validation set:')
    val_loss, val_accuracy = evaluate_model(model, val_loader)
    logger.info(f'Final Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Save final metrics
    metrics = {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'model_name': 'distilbert-base-uncased',
        'framework': 'pytorch'
    }
    save_metrics(metrics)
    
