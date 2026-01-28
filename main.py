import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import ParameterGrid, RepeatedKFold
from sklearn.metrics import accuracy_score
from pathlib import Path
import csv
from sklearn.model_selection import train_test_split
from data_preprocessing.pipeline import data_preprocessing_pipeline
from tqdm import tqdm

CSV_FILE = Path("results.csv")
csv_fields = [
    "params",
    "fold",
    "epoch",
    "val_accuracy",
    "test_accuracy",
    "train_loss",
    "val_loss",
    "learning_rate",
    "batch_size",
    "max_len"
]


_, texts, y = data_preprocessing_pipeline()

texts, _, y, _ = train_test_split(
    texts,
    y,
    train_size=0.01,
    stratify=y,
    random_state=2137
)

texts = list(texts)
y = y.reset_index(drop=True)

data_test = pd.read_csv("./data_preprocessing/data/03_test/gt_data.csv", sep=";")
y_test = data_test.iloc[:, 1].reset_index(drop=True)
X_test = data_test.iloc[:, 2].astype(str).tolist()

grid_params = {
    "max_len": [64],
    "epochs": [3],
    "batch_size": [128],
    "learning_rate": [5e-5, 3e-5, 1e-5]
}

kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2137)

if not CSV_FILE.exists():
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

device = torch.device("mps")

y = y.tolist()

for params in ParameterGrid(grid_params):
    print(f"Grid params: {params}")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"Fold: {fold_idx}")

        X_train = [texts[i] for i in train_idx]
        X_val = [texts[i] for i in val_idx]


        y_train_list = [y[i] for i in train_idx]
        y_val_list = [y[i] for i in val_idx]
        y_test_list = y_test.tolist()

        y_train = torch.tensor(y_train_list, dtype=torch.long)
        y_val = torch.tensor(y_val_list, dtype=torch.long)
        y_test_t = torch.tensor(y_test_list, dtype=torch.long)

        train_enc = tokenizer(X_train, padding=True, truncation=True, max_length=params["max_len"], return_tensors="pt")
        val_enc = tokenizer(X_val, padding=True, truncation=True, max_length=params["max_len"], return_tensors="pt")
        test_enc = tokenizer(X_test, padding=True, truncation=True, max_length=params["max_len"], return_tensors="pt")

        train_dataset = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], y_train)
        val_dataset = TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], y_val)
        test_dataset = TensorDataset(test_enc['input_ids'], test_enc['attention_mask'], y_test_t)

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=2
        )

        model.to(device)

        optimizer = AdamW(model.parameters(), lr=params["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(params["epochs"]):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc="Training"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    val_loss += loss.item()
                    val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)

            model.eval()
            test_preds = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Test"):
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            test_acc = accuracy_score(y_test, test_preds)

            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow({
                    "params": json.dumps(params),
                    "fold": fold_idx,
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                    "learning_rate": params["learning_rate"],
                    "batch_size": params["batch_size"],
                    "max_len": params["max_len"]
                })
