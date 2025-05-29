import streamlit as st
import pandas as pd

st.set_page_config(page_title="Avocado ML App", page_icon="🥑")

st.title("🥑 Avocado ML Demo")
st.markdown(
    """
    Selamat datang! Gunakan tombol di bawah untuk menjelajahi:
    - 📊 Dashboard eksplorasi data
    - 📈 Evaluasi model
    - 🔮 Prediksi tingkat kematangan
    """
)

@st.cache_data
def load_data():
    return pd.read_csv("avocado_ripeness_dataset.csv")

if "df" not in st.session_state:
    st.session_state["df"] = load_data()

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 Buka Dashboard"):
        st.switch_page("pages/1_Dashboard.py")
with col2:
    if st.button("📈 Model Performance"):
        st.switch_page("pages/2_Model_Performance.py")
with col3:
    if st.button("🔮 Prediksi"):
        st.switch_page("pages/3_Prediction.py")
