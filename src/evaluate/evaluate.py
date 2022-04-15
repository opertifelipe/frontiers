from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import pandas as pd

def check_in(row):
    if row["y_true"] in row["y_pred"]:
        return 1
    else:
        return 0

def mean_reciprocal_rank(row):
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
    
    df_evaluator["reciprocal_rank"] = df_evaluator.apply(mean_reciprocal_rank, axis=1)
    
    mean_reciprocal_rank = round(df_evaluator["reciprocal_rank"].sum() / df_evaluator.shape[0],2)

    df_evaluator["first_rank"] = df_evaluator.apply(get_first_rank, axis=1)
    
    report = classification_report(df_evaluator["y_true"], df_evaluator["first_rank"])
    
    return df_evaluator, accucary_score, mean_reciprocal_rank, report