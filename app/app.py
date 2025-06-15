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
def load_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv('../data/data.csv')
        return df
    except FileNotFoundError as e:
        st.error(f"❌ Dataset files not found: {str(e)}")
        st.error("Please ensure these files are in the same directory as your app:")
        st.code("- data.csv")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return pd.DataFrame()

@st.cache_resource(show_spinner="Loading model and vectorizer (this might take a while)...")
def load_assets():
    try:
        model = joblib.load('../models/voting_classifier.pkl')
        vectorizer = joblib.load('../models/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ File model/vectorizer tidak ditemukan: {e}")
        st.error("Pastikan file 'model.pkl' dan 'vectorizer.pkl' berada di direktori yang sama.")
        return None, None

model, vectorizer = load_assets()
df = load_dataset()
CLASS_NAMES = ['Normal', 'Depression', 'Suicidal', 'Anxiety', 'Bipolar', 'Stress', 'Personality disorder']

# 1. Menampilkan Dataset dalam bentuk Tabel (di dalam expander)
if df is not None and not df.empty:
    with st.expander("📊 Pratinjau Dataset (data.csv)"):
        st.dataframe(df)
        st.write(f"**Dimensi Data:** {df.shape[0]} baris, {df.shape[1]} kolom")

# 2. Menampilkan EDA
with st.expander("📈 Analisis Data Eksplorasi (EDA)"):
    st.markdown("#### ✏️ Tempat untuk visualisasi EDA Anda.")
    st.info("Anda bisa menambahkan berbagai plot di sini, contohnya seperti di bawah (saat ini dinonaktifkan).")
    
    # --- CONTOH TEMPLATE EDA (Anda bisa mengaktifkan dan mengubah ini) ---
    # st.write("**Distribusi Label Kondisi Mental**")
    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.countplot(data=df, y='label', ax=ax, order=df['label'].value_counts().index)
    # ax.set_title('Distribusi Kelas')
    # ax.set_xlabel('Jumlah')
    # ax.set_ylabel('Kelas')
    # st.pyplot(fig)

st.markdown("---")

st.header("🔮 Coba Klasifikasi Teks Anda")

user_text = st.text_area(
    "Masukkan teks di sini untuk dianalisis:",
    placeholder="Contoh: I feel so empty and have no motivation to do anything anymore...",
    height=150
)

if st.button("🚀 Analisis Sekarang", type="primary"):
    if model and vectorizer:
        if user_text:
            # --- Proses Backend ---
            # Preprocess teks input
            preprocessed_text = preprocess_text(user_text)
            
            # Vektorisasi teks
            vectorized_text = vectorizer.transform([preprocessed_text])
            
            # Prediksi dengan model
            prediction_proba = model.predict_proba(vectorized_text)
            prediction_index = np.argmax(prediction_proba)
            predicted_class = CLASS_NAMES[prediction_index]
            confidence = prediction_proba[0][prediction_index]
            
            # --- Tampilan Output ---
            st.subheader("✅ Hasil Analisis")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric("Predicted Class", predicted_class)
                st.metric("Confidence Score", f"{confidence:.2%}")

                # Plot probabilitas
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(CLASS_NAMES, prediction_proba[0], color='skyblue')
                ax.set_xlabel('Probabilitas')
                ax.set_title('Distribusi Probabilitas Prediksi')
                ax.set_xlim(0, 1)
                # Highlight bar prediksi
                bars[prediction_index].set_color('salmon')
                st.pyplot(fig)

            with col2:
                st.info("**Detail Proses:**")
                st.write("**Teks Asli:**")
                st.write(f"`{user_text}`")
                st.write("**Teks Setelah Preprocessing:**")
                st.write(f"`{preprocessed_text}`")
                st.write("**Output TF-IDF Vectorizer (Representasi Numerik):**")
                st.write(f"- **Dimensi Vektor:** {vectorized_text.shape}")
                st.write(f"- **Jumlah Fitur Non-Zero:** {vectorized_text.nnz}")
                # Menampilkan beberapa elemen dari sparse matrix untuk ilustrasi
                st.code(f"{vectorized_text.toarray()[:, :15]} ...", language='text')

        else:
            st.warning("⚠️ Mohon masukkan teks untuk dianalisis.")
    else:
        st.error("Model tidak dapat dimuat. Aplikasi tidak dapat melakukan prediksi.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><small>Mental Illness Classification App | Dibuat dengan Streamlit</small></p>
</div>
""", unsafe_allow_html=True)