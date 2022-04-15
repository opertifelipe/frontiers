import numpy as np
from utils.utils import nlp

def get_embeddings_keyword(list_of_keywords):
    vectors = []
    for key in list_of_keywords:
        key_doc = nlp(key)
        vectors.append(key_doc.vector)
    return np.array(vectors).mean(axis=0)

def create_embeddings_keywords(df):
    df["embeddings"] = df["keywords_cleaned"].parallel_apply(get_embeddings_keyword)
    return df    