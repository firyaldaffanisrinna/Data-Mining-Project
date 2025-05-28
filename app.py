import streamlit as st
import pandas as pd

st.set_page_config(page_title="Avocado ML App", page_icon="🥑")

st.title("🥑 Avocado ML Demo")
st.markdown(
    """
    Selamat datang! Gunakan menu sidebar untuk:
    - **Dashboard** – eksplorasi data.
    - **Model Performance** – lihat metrik evaluasi model.
    - **Prediction** – demo prediksi online.
    """
)

# Cache dataset agar tidak dibaca berulang-ulang
@st.cache_data
def load_data():
    return pd.read_csv("avocado_ripeness_dataset.csv")

st.session_state["df"] = load_data()

st.success("Dataset dimuat. Silakan pilih halaman di sidebar.")
