from src.utils.utils import IO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

journal_embeddings_keywords_word2vec = IO(filename="journals_embeddings_keywords_word2vec",folder="04_model",format_="pickle").load()
journal_embeddings_keywords_tfidf = IO(filename="journals_embeddings_keywords_tfidf",folder="04_model",format_="pickle").load()
journal_embeddings_keywords_sbert = IO(filename="journals_embeddings_keywords_sbert",folder="04_model",format_="pickle").load()

journal_embeddings_document_word2vec = IO(filename="journals_embeddings__document_word2vec",folder="04_model",format_="pickle").load()
journal_embeddings_document_tfidf = IO(filename="journals_embeddings__document_tfidf",folder="04_model",format_="pickle").load()
journal_embeddings_document_sbert = IO(filename="journals_embeddings__document_sbert",folder="04_model",format_="pickle").load()


def predict(df, embeddings_function, embedding_type, journal_embeddings, tf_idf_training=True):
    df = embeddings_function(df, embedding_type, tf_idf_training)
    df = df.reset_index(drop=True)
    predictions = []
    for index, row in df.iterrows():
        cosine_sim = cosine_similarity(row["embeddings"].reshape(1,-1), 
                                       np.stack(journal_embeddings["embeddings"].values))
        idx = (-cosine_sim[0]).argsort()[:3]
        prediction = [journal_embeddings["journal"][idx[0]], 
                      journal_embeddings["journal"][idx[1]], 
                      journal_embeddings["journal"][idx[2]]]
        predictions.append(prediction)
    df["prediction"] = predictions
    return df

def predict_keyword_word2vec(df_test):    
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_keywords,
                            embedding_type="word2vec",
                            journal_embeddings=journal_embeddings_keywords_word2vec)
    return df_evaluation

def predict_keyword_tfidf(df_test):
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_keywords,
                            embedding_type="tfidf",
                            tf_idf_training=False,
                            journal_embeddings=journal_embeddings_keywords_tfidf)
    return df_evaluation


def predict_keyword_sbert(df_test):
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_keywords,
                            embedding_type="sbert",
                            journal_embeddings=journal_embeddings_keywords_sbert)

    return df_evaluation



def predict_document_word2vec(df_test):
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_document,
                            embedding_type="word2vec",
                            journal_embeddings=journal_embeddings_document_word2vec)
    return df_evaluation


def predict_document_tfidf(df_test):
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_document,
                            embedding_type="tfidf",
                            tf_idf_training=False,
                            journal_embeddings=journal_embeddings_document_tfidf)
    return df_evaluation

def predict_document_sbert(df_test):
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_document,
                            embedding_type="sbert",
                            journal_embeddings=journal_embeddings_document_sbert)
    return df_evaluation

