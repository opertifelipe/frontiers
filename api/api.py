from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from api.core.predict import predict_document_sbert

import pandas as pd
 
class ScientificPaper(BaseModel):
    text: str


app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "alive"}

@app.post("/journal/recommendation")
async def journal_recommendation(paper: ScientificPaper):
    df = pd.DataFrame([[paper.text]], columns=["text"])
    df_predicted = await predict_document_sbert(df)
    return {"journals_recommendation":df_predicted["prediction"].tolist()[0]}