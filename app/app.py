import streamlit as st
import pandas as pd
import numpy as np
import fitz
import requests
import json

st.set_page_config(layout="wide")


st.title('Frontiers Journal Recommendation')

upload_insert = st.selectbox("Would you prefer to upload a pdf paper or insert a text", 
            ["UPLOAD","INSERT"], 
            index=1)

text = None

if upload_insert == "INSERT":
    text = st.text_area("Insert here the text of the paper", value="")
else:
    uploaded_file = st.file_uploader("Upload here the pdf of the paper", 
                     type="pdf", 
                     accept_multiple_files=False)
    if uploaded_file:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()

model = st.selectbox(
     "Select the model",
     ('SBERT', 'TFIDF', 'Word2Vec'))

embedding_type = st.radio("Select the embedding type", 
                                ["All document","Only keywords"], 
                                index=0)     

mapping_model = {'SBERT':'sbert', 'TFIDF':'tfidf', 'Word2Vec':'word2vec'}
mapping_embedding_type = {'All document':'document', 'Only keywords':'keywords'}

if text:
    url = "http://127.0.0.1:8082/journal/recommendation"
    payload = {"text":text, 
               "model":mapping_model[model], 
               "embedding_type":mapping_embedding_type[embedding_type]}


    response = requests.request("POST", url, data=json.dumps(payload)).json()
    st.markdown(f"""
    **1. Recommendation**: {response["journals_recommendation"][0]}

    **2. Recommendation**: {response["journals_recommendation"][1]}

    **3. Recommendation**: {response["journals_recommendation"][2]}""", unsafe_allow_html=False)






