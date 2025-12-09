import random
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from data_preprocessing.pipeline import data_preprocessing_pipeline
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert encoding values and labels to torch tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def train_bert(X_texts=None, y=None, max_len=128, epochs=2, batch_size=16, device=None):
    # Set device to GPU if available, otherwise CPU
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load data from pipeline if no input is provided
    if X_texts is None or y is None:
        _, X_texts, y = data_preprocessing_pipeline()

    # Ensure X_texts is a list of strings
    X_texts = list(X_texts)

    # Split into train and test sets
    texts_train, texts_test, y_train, y_test = train_test_split(
        X_texts, y, test_size=0.2, stratify=y, random_state=42
    )


    # Tokenization with BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_enc = tokenizer(texts_train, truncation=True, padding=True, max_length=max_len)
    test_enc = tokenizer(texts_test, truncation=True, padding=True, max_length=max_len)


    # Create Dataset and DataLoader
    train_dataset = TextDataset(train_enc, y_train.values)
    test_dataset = TextDataset(test_enc, y_test.values)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

  
    # Load pretrained BERT
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Fine-tuning
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()


    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print metrics
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("F1-score:", f1_score(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(all_labels, all_preds))


    # Save model and tokenizer
    model.save_pretrained("bert_model/")
    tokenizer.save_pretrained("bert_model/")

    return model
