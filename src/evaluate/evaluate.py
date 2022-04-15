from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from utils.utils import IO
from train.keywords_approach import create_embeddings_keywords
from train.document_approach import create_embeddings_document

def check_in(row):
    if row["y_true"] in row["y_pred"]:
        return 1
    else:
        return 0

def calculate_mean_reciprocal_rank(row):
    if row["y_true"] in row["y_pred"]:
        index = row["y_pred"].index(row["y_true"])
        return 1/(index+1)
    else:
        return 0

def get_first_rank(row):
    if row["y_true"] in row["y_pred"]:
        return row["y_true"]
    else:
        return row["y_pred"][0]


def evaluator(y_true, y_pred):
    df_evaluator = pd.DataFrame(y_true,columns=["y_true"])
    df_evaluator["y_pred"] = y_pred
    
    df_evaluator["tp"] = df_evaluator.apply(check_in, axis=1)
    accucary_score = round(df_evaluator["tp"].sum() / df_evaluator.shape[0], 2)
    
    df_evaluator["reciprocal_rank"] = df_evaluator.apply(calculate_mean_reciprocal_rank, axis=1)
    
    mean_reciprocal_rank = round(df_evaluator["reciprocal_rank"].sum() / df_evaluator.shape[0],2)

    df_evaluator["first_rank"] = df_evaluator.apply(get_first_rank, axis=1)
    
    report = classification_report(df_evaluator["y_true"], df_evaluator["first_rank"])
    
    return df_evaluator, accucary_score, mean_reciprocal_rank, report

def generate_evaluation_report( y_true, y_pred):
    _, accuracy_score, mean_reciprocal_rank, report = evaluator(y_true, y_pred)
    eval_report = {"accuracy_total":accuracy_score,
                   "mean_reciprocal_rank":mean_reciprocal_rank,
                   "precision_recall_f1score":report}
    return eval_report

def generate_baseline_evaluation(df_test):
    high_frequency_solution = []

    for i in range(df_test.shape[0]):
        high_frequency_solution.append(["Frontiers in Microbiology",
                                        "Frontiers in Psychology",
                                        "Frontiers in Immunology"])
    eval_report = generate_evaluation_report(df_test["journal"].tolist(),high_frequency_solution)                                  
    IO(eval_report, filename="evaluation_baseline",folder="05_report",format_="json").save()


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

def evaluate_keyword_word2vec(df_test):
    journal_embeddings = IO(filename="journals_embeddings_keywords_word2vec",folder="04_model",format_="pickle").load()
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_keywords,
                            embedding_type="word2vec",
                            journal_embeddings=journal_embeddings)
    
    evaluation = generate_evaluation_report(df_evaluation["journal"].tolist(),
                                           df_evaluation["prediction"].tolist())
    IO(evaluation, filename="evaluation_keywords_word2vec",folder="05_report",format_="json").save()

def evaluate_keyword_tfidf(df_test):
    journal_embeddings = IO(filename="journals_embeddings_keywords_tfidf",folder="04_model",format_="pickle").load()
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_keywords,
                            embedding_type="tfidf",
                            tf_idf_training=False,
                            journal_embeddings=journal_embeddings)
    
    evaluation = generate_evaluation_report(df_evaluation["journal"].tolist(),
                                           df_evaluation["prediction"].tolist())
    IO(evaluation, filename="evaluation_keywords_tfidf",folder="05_report",format_="json").save()

def evaluate_document_word2vec(df_test):
    journal_embeddings = IO(filename="journals_embeddings_document_word2vec",folder="04_model",format_="pickle").load()
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_document,
                            embedding_type="word2vec",
                            journal_embeddings=journal_embeddings)
    
    evaluation = generate_evaluation_report(df_evaluation["journal"].tolist(),
                                           df_evaluation["prediction"].tolist())
    IO(evaluation, filename="evaluation_document_word2vec",folder="05_report",format_="json").save()

def evaluate_document_tfidf(df_test):
    journal_embeddings = IO(filename="journals_embeddings_document_tfidf",folder="04_model",format_="pickle").load()
    df_evaluation = predict(df=df_test, 
                            embeddings_function=create_embeddings_document,
                            embedding_type="tfidf",
                            tf_idf_training=False,
                            journal_embeddings=journal_embeddings)
    
    evaluation = generate_evaluation_report(df_evaluation["journal"].tolist(),
                                           df_evaluation["prediction"].tolist())
    IO(evaluation, filename="evaluation_document_tfidf",folder="05_report",format_="json").save()