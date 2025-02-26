import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="Crypto Analysis Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def apply_theme():
    set_page_config()
    load_css()
