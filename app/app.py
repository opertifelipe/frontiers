import streamlit as st
import pandas as pd
import numpy as np
import fitz

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
                                index=1)     





