from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from api.core.predict import (predict_keyword_word2vec, 
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

pandarallel.initialize(nb_workers=1, progress_bar=True)

class ScientificPaper(BaseModel):
    text: str
    model: str
    embedding_type: str


app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "alive"}


@app.post("/journal/recommendation")
async def journal_recommendation(paper: ScientificPaper):
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

    return {"journals_recommendation":df_predicted["prediction"].tolist()[0]}