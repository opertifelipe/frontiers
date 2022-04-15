import numpy as np
from utils.utils import nlp

def get_embeddings_document_word2vec(text):
    doc = nlp(text)
    return doc.vector

def create_embeddings_document(df, embedding_type):
    if embedding_type == "word2vec":
        df["embeddings"] = df["preprocessed_text"].parallel_apply(get_embeddings_document_word2vec)
    return df    