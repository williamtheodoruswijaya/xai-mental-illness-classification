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
def load_dataset():
    base = os.getcwd()
    for root, dirs, files in os.walk(base):
        if "data.csv" in files:
            df = pd.read_csv(os.path.join(root, "data.csv"), encoding='utf-8')
            return df
    st.error("❌ data.csv tidak ditemukan di seluruh project!")
    return pd.DataFrame()

st.title("🧠 Mental Illness Classification App")
st.markdown("Analyze text for emotional content and intensity using machine learning models")

with st.spinner("🔄 Memuat data dan UI..."):
    df = load_dataset()

if df is not None and not df.empty:
    with st.expander("📊 Dataset Overview (data.csv)"):
        st.dataframe(df)
        st.write(f"**Dimensi Data:** {df.shape[0]} baris, {df.shape[1]} kolom")

with st.expander("📈 Exploratory Data Analysis (EDA)"):
    st.markdown("#### 📊 Distribusi Label Kondisi Mental")
    if 'status' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y='status', data=df, order=df['status'].value_counts().index, ax=ax, hue='status', palette="Set2")
        ax.set_title("Distribusi Kelas")
        ax.set_xlabel("Jumlah")
        ax.set_ylabel("Status")
        st.pyplot(fig)
    else:
        st.warning("Kolom 'status' tidak ditemukan pada dataset.")

    st.markdown("#### ☁️ WordCloud per Label (opsional)")
    if 'status' in df.columns and not df.empty:
        selected_label = st.selectbox("Pilih label untuk melihat WordCloud", sorted(df['status'].dropna().unique()))
        if selected_label:
            label_texts = df[df['status'] == selected_label]['statement'].dropna().astype(str)
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

            response = requests.post(API_URL, json={"input": [preprocessed_text]}, timeout=5)
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
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal menghubungi API. Coba lagi: {e}")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
    else:
        st.warning("⚠️ Mohon masukkan teks untuk dianalisis.")

st.markdown("---")
st.header("🔍 Penjelasan Prediksi dengan LIME")

user_text_explain = st.text_area(
    "Masukkan teks untuk dijelaskan menggunakan LIME:",
    placeholder="Contoh: im so stress",
    height=150,
    key="lime_input"
)

if st.button("🧠 Jalankan LIME Explanation", type="secondary"):
    if user_text_explain:
        try:
            API_EXPLAIN = "https://adamantix-ensemble-model-mental-illness-classification.hf.space/mic-explain"
            response = requests.post(API_EXPLAIN, json={"input": user_text_explain}, timeout=5)
            response.raise_for_status()
            result = response.json()

            st.success(f"✅ Prediksi Model: **{result['prediction']}**")

            # Probabilities
            st.subheader("📊 Probabilitas Kelas")
            fig, ax = plt.subplots(figsize=(8, 4))
            probs = result["probabilities"]
            ax.barh(list(probs.keys()), list(probs.values()), color='lightgreen')
            ax.set_xlabel("Probabilitas")
            ax.set_title("Distribusi Probabilitas")
            st.pyplot(fig)

            # Explanation
            st.subheader("🧠 Penjelasan Kata-kata Penting (LIME)")
            explanation = result.get("explanation", [])
            if explanation:
                exp_words = [item["word"] for item in explanation]
                exp_weights = [item["weight"] for item in explanation]

                # Highlight positive/negative contribution
                colors = ['salmon' if w < 0 else 'skyblue' for w in exp_weights]

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.bar(exp_words, exp_weights, color=colors)
                ax2.axhline(0, color='gray', linestyle='--')
                ax2.set_ylabel("Bobot")
                ax2.set_title("Kontribusi Kata terhadap Prediksi")
                st.pyplot(fig2)

                # Optional: Word contribution text
                st.markdown("**Detail Kontribusi Kata:**")
                for item in explanation:
                    emoji = "🔻" if item["weight"] < 0 else "🔺"
                    st.markdown(f"{emoji} **{item['word']}** → `weight: {item['weight']:.4f}`")
            else:
                st.warning("Tidak ada penjelasan yang tersedia dari model.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal menghubungi API LIME: {e}")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu untuk dianalisis.")


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><small>Mental Illness Classification App | Dibuat dengan Streamlit</small></p>
</div>
""", unsafe_allow_html=True)
