from data_preprocessing.pipeline import data_preprocessing_pipeline


# X: scipy.sparse.csr_matrix, y: pd.Series
X, y = data_preprocessing_pipeline()