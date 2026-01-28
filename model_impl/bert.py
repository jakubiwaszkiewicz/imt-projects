import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import BertTokenizer, TFBertForSequenceClassification
from data_preprocessing.pipeline import data_preprocessing_pipeline

def train_bert(X_texts, y, max_len=128, epochs=2, batch_size=16, learning_rate=0.001):
    # Wczytanie danych z pipeline jeśli X_texts i y są None
    if X_texts is None or y is None:
        _, X_texts, y = data_preprocessing_pipeline()

    # Upewniamy się, że X_texts to lista stringów
    X_texts = list(X_texts)

    # Podział train/test
    texts_train, texts_test, y_train, y_test = train_test_split(
        X_texts, y, test_size=0.2, stratify=y, random_state=42
    )

    # Tokenizacja BERT
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")


    train_enc = tokenizer(
        texts_train,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="tf"
    )

    test_enc = tokenizer(
        texts_test,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="tf"
    )

    # Przygotowanie datasetów TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_enc),
        y_train.values
    )).shuffle(1000).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_enc),
        y_test.values
    )).batch(batch_size)

    # Wczytanie pretrained BERT
    model = TFBertForSequenceClassification.from_pretrained(
    "google-bert/bert-base-multilingual-cased",
        from_pt=True
    )

    # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Fine-tuning
    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)

    # Ewaluacja
    logits = model.predict(test_dataset).logits
    y_pred = tf.argmax(logits, axis=1).numpy()

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Zapis modelu
    model.save_pretrained("bert_model/")
    tokenizer.save_pretrained("bert_model/")


    # Wyniki
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)

    logits = model.predict(test_dataset).logits
    y_pred = tf.argmax(logits, axis=1).numpy()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results = {
        "accuracy": acc,
        "f1_score": f1,
        "train_loss": history.history["loss"][-1],
        "val_loss": history.history.get("val_loss", [None])[-1],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_len": max_len
    }

    return model, results

if __name__ == "__main__":
    train_bert(None, None)
