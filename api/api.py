from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from src.evaluate.evaluate import (predict_keyword_word2vec, 
                                   predict_keyword_tfidf,
                                   predict_keyword_sbert,
                                   predict_document_word2vec,
                                   predict_document_tfidf,
                                   predict_document_sbert)

from src.preprocess.preprocess import preprocess

import pandas as pd

from pandarallel import pandarallel

import warnings
warnings.filterwarnings("ignore")

pandarallel.initialize(progress_bar=True)

class ScientificPaper(BaseModel):
    text: str
    model: str
    embedding_type: str


app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "alive"}


@app.post("/journal/recommendation")
def create_item(paper: ScientificPaper):
    df = pd.DataFrame([[paper.text]], columns=["text"])
    df = preprocess(df)
    if paper.embedding_type == "keywords":
        if paper.model == "word2vec":
            df_predicted = predict_keyword_word2vec(df)
        elif paper.model == "tfidf":
            df_predicted = predict_keyword_tfidf(df)
        elif paper.model == "sbert":
            df_predicted = predict_keyword_sbert(df)
    elif paper.embedding_type == "document":
        if paper.model == "word2vec":
            df_predicted = predict_document_word2vec(df)
        elif paper.model == "tfidf":
            df_predicted = predict_document_tfidf(df)
        elif paper.model == "sbert":
            df_predicted = predict_document_sbert(df)

    return {"journal":df_predicted["prediction"].tolist()}