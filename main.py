from data_preprocessing.pipeline import data_preprocessing_pipeline
from model_impl.bert import train_bert

# X: scipy.sparse.csr_matrix, y: pd.Series
X, X_texts, y = data_preprocessing_pipeline()

# Train BERT model on the provided texts and labels
# train_bert(X_texts, y)