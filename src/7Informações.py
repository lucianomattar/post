import streamlit as st
from pathlib import Path
import base64

def run():

    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()

    intro_markdown = read_markdown_file("info.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
