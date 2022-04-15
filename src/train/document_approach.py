import numpy as np
from utils.utils import nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.utils import IO

def get_embeddings_document_word2vec(text):
    doc = nlp(text)
    return doc.vector

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

    return df    