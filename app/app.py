import streamlit as st

st.set_page_config(
    page_title="Mental Illness Classification App",
    page_icon="🧠",
    layout="wide"
)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import string
import requests
from wordcloud import WordCloud

# Back-end
CLASS_NAMES = [
    'Normal',
    'Depression',
    'Suicidal',
    'Anxiety',
    'Bipolar',
    'Stress',
    'Personality disorder',
]

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
def load_dataset() -> pd.DataFrame:
    try:
        BASE_DIR = os.path.dirname(__file__)
        DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'data.csv')
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError as e:
        st.error(f"❌ Dataset files not found: {str(e)}")
        st.error("Please ensure these files are in the same directory as your app:")
        st.code("- data.csv")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return pd.DataFrame()

df = load_dataset()

st.title("Mental Ilness Classification App")
st.markdown("Analyze text for emotional content and intensity using machine learning models")

if df is not None and not df.empty:
    with st.expander("📊 Pratinjau Dataset (data.csv)"):
        st.dataframe(df)
        st.write(f"**Dimensi Data:** {df.shape[0]} baris, {df.shape[1]} kolom")

with st.expander("📈 Analisis Data Eksplorasi (EDA)"):
    st.markdown("#### 📊 Distribusi Label Kondisi Mental")
    if 'status' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y='status', data=df, order=df['status'].value_counts().index, ax=ax, palette="Set2")
        ax.set_title("Distribusi Kelas")
        ax.set_xlabel("Jumlah")
        ax.set_ylabel("Status")
        st.pyplot(fig)
    else:
        st.warning("Kolom 'status' tidak ditemukan pada dataset.")

    st.markdown("#### ☁️ WordCloud per Label (opsional)")
    selected_label = st.selectbox("Pilih label untuk melihat WordCloud", sorted(df['status'].unique()))
    if selected_label:
        label_texts = df[df['status'] == selected_label]['text'].dropna().astype(str)
        combined_text = ' '.join(label_texts)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(combined_text)
        
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title(f'WordCloud untuk Label: {selected_label}', fontsize=16)
        st.pyplot(fig_wc)

st.markdown("---")

st.header("🔮 Coba Klasifikasi Teks Anda")

user_text = st.text_area(
    "Masukkan teks di sini untuk dianalisis:",
    placeholder="Contoh: I feel so empty and have no motivation to do anything anymore...",
    height=150
)

if st.button("🚀 Analisis Sekarang", type="primary"):
    if user_text:
        try:
            API_URL = "https://adamantix-ensemble-model-mental-illness-classification.hf.space/mic-predict-many"
            preprocessed_text = preprocess_text(user_text)

            response = requests.post(API_URL, json={"input": [preprocessed_text]})
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and result:
                prediction_proba = result[0]
                prediction_index = max(prediction_proba, key=prediction_proba.get)
                confidence = prediction_proba[prediction_index]

                st.subheader("✅ Hasil Analisis")
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.metric("Predicted Class", prediction_index)
                    st.metric("Confidence Score", f"{confidence:.2%}")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    probs = [prediction_proba[cls] for cls in CLASS_NAMES]
                    bars = ax.barh(CLASS_NAMES, probs, color='skyblue')
                    ax.set_xlabel('Probabilitas')
                    ax.set_title('Distribusi Probabilitas Prediksi')
                    ax.set_xlim(0, 1)
                    bars[CLASS_NAMES.index(prediction_index)].set_color('salmon')
                    st.pyplot(fig)

                with col2:
                    st.info("**Detail Proses:**")
                    st.write("**Teks Asli:**")
                    st.write(f"`{user_text}`")
                    st.write("**Teks Setelah Preprocessing:**")
                    st.write(f"`{preprocessed_text}`")

            else:
                st.error("❌ Format respons tidak sesuai.")
        except Exception as e:
            st.error(f"❌ Error API: {e}")
    else:
        st.warning("⚠️ Mohon masukkan teks untuk dianalisis.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><small>Mental Illness Classification App | Dibuat dengan Streamlit</small></p>
</div>
""", unsafe_allow_html=True)
