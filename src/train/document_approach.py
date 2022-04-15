import numpy as np
from src.utils.utils import nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.utils import IO
from sentence_transformers import SentenceTransformer
from functools import partial
import torch

def get_embeddings_document_word2vec(text):
    doc = nlp(text)
    return doc.vector

def get_embeddings_document_sbert(text, model):
    sentences = text.split(". ")
    embeddings = model.encode(sentences)
    return np.mean(embeddings, axis=0)

def create_embeddings_document(df, embedding_type, tf_idf_training = True):
    if embedding_type == "word2vec":
        #df["embeddings"] = df["preprocessed_text"].parallel_apply(get_embeddings_document_word2vec)
        df["embeddings"] = df["text"].parallel_apply(get_embeddings_document_word2vec)
    elif embedding_type == "tfidf":
        if tf_idf_training:
            vectorizer = TfidfVectorizer(max_features=10000)
            # df["embeddings"] = list(vectorizer.fit_transform(df["preprocessed_text"].values).toarray())
            df["embeddings"] = list(vectorizer.fit_transform(df["text"].values).toarray())    
            IO(vectorizer, "journals_embeddings_document_vectorizer_tfidf","04_model","pickle").save()
        else:
            vectorizer = IO(filename="journals_embeddings_document_vectorizer_tfidf",folder="04_model",format_="pickle").load()
            # df["embeddings"] = list(vectorizer.transform(df["preprocessed_text"].values).toarray())   
            df["embeddings"] = list(vectorizer.transform(df["text"].values).toarray())   
    elif embedding_type == "sbert":
        if torch.cuda.is_available():
            model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
        else:
            model = SentenceTransformer("all-mpnet-base-v2")                        
        df["embeddings"] = df["text"].apply(partial(get_embeddings_document_sbert, model=model))

    return df    