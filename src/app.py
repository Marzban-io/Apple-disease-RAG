"""Phase 8: a simple web app for the apple disease assistant.
   Run from the project folder:  streamlit run src/app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from answer import build_prompt, generate, retrieve   # reuse the Phase 7 functions

st.set_page_config(page_title="Apple Disease Assistant", page_icon="🍎")
st.title("🍎 Apple Disease Assistant")
st.caption("Answers come only from FAO reports and orchard guides, with page references. "
           "Always confirm treatments with a local agronomist.")


@st.cache_resource
def get_client():
    """Create the Gemini client once and reuse it for every question."""
    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


question = st.text_input("Your question", placeholder="Which apple diseases spread with rain and high humidity?")

if st.button("Ask") and question.strip():
    try:
        with st.spinner("Searching the documents and writing the answer..."):
            chunks = retrieve(question)                                # R
            answer = generate(get_client(), build_prompt(question, chunks))   # A + G
    except Exception as error:
        st.error(f"Something went wrong: {error}")
        st.stop()

    st.markdown(answer)
    st.subheader("Sources")
    for i, (text, meta) in enumerate(chunks, start=1):
        with st.expander(f"[{i}] {meta['source']}, page {meta['page']}"):
            st.write(text)