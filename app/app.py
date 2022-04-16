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

if st.button('Find recommended journals'):
    url = "http://api:8086/journal/recommendation"
    payload = {"text":text}
    
    response = requests.request("POST", url, data=json.dumps(payload)).json()
    st.markdown(f"""
    **Recommendation:**: 
    
    1. *{response["journals_recommendation"][0]}*
    2. *{response["journals_recommendation"][1]}*    
    3. *{response["journals_recommendation"][2]}*""", 
    
    unsafe_allow_html=False)






