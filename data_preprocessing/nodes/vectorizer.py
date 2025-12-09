import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def vectorizer(data: pd.DataFrame) -> tuple[scipy.sparse.csr_matrix, pd.Series]:
    X_texts = data["post_text"].tolist()

    vec = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=5,  # have to be at least in >= 5 posts
        max_df=0.8,  # skip, if in >80%
        stop_words='english'
    )
    
    X = vec.fit_transform(data["post_text"])
    y = data["is_russian_disinformation"].astype(int)

    return X, X_texts, y

