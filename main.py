import pandas as pd
import json
import tensorflow as tf
import csv
from pathlib import Path
from sklearn.model_selection import ParameterGrid, RepeatedKFold
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing.pipeline import data_preprocessing_pipeline
from model_impl.bert import train_bert
from transformers import BertTokenizer

CSV_FILE = Path("results.csv")
csv_fields = [
    "params", "fold", "epoch", "val_accuracy", "val_f1",
    "test_accuracy", "test_f1", "train_loss", "val_loss",
    "learning_rate", "batch_size", "max_len"
]


_, texts, y = data_preprocessing_pipeline()

texts = texts[::100]
y = y[::100].reset_index(drop=True)

data_test = pd.read_csv("./data_preprocessing/data/03_test/gt_data.csv", sep=";")
y_test = data_test.iloc[:, 1].reset_index(drop=True)
X_test_texts = data_test.iloc[:, 2].reset_index(drop=True)

print(data_test)
print(y_test)
print(type(X_test_texts))

grid_params = {
    "max_len": [64],
    "epochs": [5],
    "batch_size": [128],
    "learning_rate": [0.001, 0.0001]
}


kf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=42)


if not CSV_FILE.exists():
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()


for params in ParameterGrid(grid_params):
    print(f"Grid params: {params}")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"Fold: {fold_idx}")

        texts_train = [texts[i] for i in train_idx]
        texts_val = [texts[i] for i in val_idx]
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        model, results = train_bert(
            X_texts=texts_train,
            y=y_train,
            max_len=params["max_len"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"]
        )

        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        val_enc = tokenizer(
            texts_val,
            truncation=True,
            padding=True,
            max_length=params["max_len"],
            return_tensors="tf"
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(dict(val_enc)).batch(params["batch_size"])
        val_logits = model.predict(val_dataset).logits
        y_val_pred = tf.argmax(val_logits, axis=1).numpy()
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)

        # Ewaluacja na zbiorze testowym polityków
        test_enc = tokenizer(
            list(X_test_texts),
            truncation=True,
            padding=True,
            max_length=params["max_len"],
            return_tensors="tf"
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(dict(test_enc)).batch(params["batch_size"])
        test_logits = model.predict(test_dataset).logits
        y_test_pred = tf.argmax(test_logits, axis=1).numpy()
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Zapis do CSV
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({
                "params": json.dumps(params),
                "fold": fold_idx,
                "epoch": params["epochs"],
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "train_loss": results["train_loss"],
                "val_loss": results["val_loss"],
                "learning_rate": results["learning_rate"],
                "batch_size": results["batch_size"],
                "max_len": results["max_len"]
            })
