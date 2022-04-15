import streamlit as st
import pandas as pd
import numpy as np

st.title('Frontiers Journal Recommendation')

upload_insert = st.selectbox("Would you prefer to upload a pdf paper or insert a text", 
            ["UPLOAD","INSERT"], 
            index=1)

if upload_insert == "INSERT":
    st.text_area("Insert here the text of the paper", value="")
else:
    st.file_uploader("Upload here the pdf of the paper", type="pdf", accept_multiple_files=False)
