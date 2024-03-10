import streamlit as st
from pathlib import Path
import base64
from streamlit_extras.app_logo import add_logo

def run():

    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()

    intro_markdown = read_markdown_file("o_que_e_ona.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)