import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import joblib
import string

st.set_page_config(
    page_title="Mental Illness Classification App",
    page_icon="🧠",
    layout="wide"
)

st.title("Mental Ilness Classification App")
st.markdown("Analyze text for emotional content and intensity using machine learning models")

# config smping
st.sidebar.header("⚙️ Settings")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@st.cache_data
def load_datasets():
    try:
        df = pd.read_csv('../data/data.csv')
        return df, len(df)
    except FileNotFoundError as e:
        st.error(f"❌ Dataset files not found: {str(e)}")
        st.error("Please ensure these files are in the same directory as your app:")
        st.code("- data.csv")
        return None, 0
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return None, 0