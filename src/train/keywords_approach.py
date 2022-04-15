import numpy as np
from utils.utils import nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.utils import IO
from sentence_transformers import SentenceTransformer
from functools import partial


def get_embeddings_keyword_word2vec(list_of_keywords):
    vectors = []
    for key in list_of_keywords:
        key_doc = nlp(key)
        vectors.append(key_doc.vector)
    return np.array(vectors).mean(axis=0)

def get_sentence_keyword_tfidf(list_of_keywords):
    return " ".join(list_of_keywords)

def get_embeddings_keyword_sbert(model, list_of_keywords):
    embeddings = model.encode(list_of_keywords)
    doc_embeddings_bert = np.mean(embeddings, axis=0)
    return doc_embeddings_bert

def create_embeddings_keywords(df, embedding_type, tf_idf_training = True):
    if embedding_type == "word2vec":
        df["embeddings"] = df["keywords_cleaned"].parallel_apply(get_embeddings_keyword_word2vec)
    elif embedding_type == "tfidf":
        if tf_idf_training:
            vectorizer = TfidfVectorizer(max_features=5000)
            df["sentence"] = df["keywords_cleaned"].parallel_apply(get_sentence_keyword_tfidf)
            df["embeddings"] = list(vectorizer.fit_transform(df["sentence"].values).toarray())   
            IO(vectorizer, "journals_embeddings_keywords_vectorizer_tfidf","04_model","pickle").save()
        else:
            vectorizer = IO(filename="journals_embeddings_keywords_vectorizer_tfidf",folder="04_model",format_="pickle").load()
            df["sentence"] = df["keywords_cleaned"].parallel_apply(get_sentence_keyword_tfidf)
            df["embeddings"] = list(vectorizer.transform(df["sentence"].values).toarray())   
    elif embedding_type == "sbert":
        model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
        df["embeddings"] = df["keywords_cleaned"].apply(partial(get_embeddings_keyword_sbert, model=model))
    return df    