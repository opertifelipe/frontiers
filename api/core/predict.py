from api.utils.utils import IO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import partial
import torch
 
journal_embeddings_document_sbert = IO(filename="journals_embeddings_document_sbert",folder="04_model",format_="pickle").load()

if torch.cuda.is_available():
    model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
else:
    model = SentenceTransformer("all-mpnet-base-v2")

async def create_embeddings_document(df):
    def get_embeddings_document_sbert(text, model):
        sentences = text.split(". ")
        embeddings = model.encode(sentences)
        return np.mean(embeddings, axis=0)                        
    df["embeddings"] = df["text"].apply(partial(get_embeddings_document_sbert, model=model))
    return df    

async def predict_document_sbert(df):
    df = await create_embeddings_document(df)
    df = df.reset_index(drop=True)
    predictions = []
    for index, row in df.iterrows():
        cosine_sim = cosine_similarity(row["embeddings"].reshape(1,-1), 
                                       np.stack(journal_embeddings_document_sbert["embeddings"].values))
        idx = (-cosine_sim[0]).argsort()[:3]
        prediction = [journal_embeddings_document_sbert["journal"][idx[0]], 
                      journal_embeddings_document_sbert["journal"][idx[1]], 
                      journal_embeddings_document_sbert["journal"][idx[2]]]
        predictions.append(prediction)
    df["prediction"] = predictions
    return df

