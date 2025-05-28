import streamlit as st

st.title("🔍 Prediksi (Demo)")

st.write("Fitur ini masih dalam pengembangan.")
label = st.selectbox("Pilih jenis tumor (dummy input):", ["brain_glioma", "brain_menin", "brain_tumor"])

if st.button("Prediksi"):
    st.success(f"Model memprediksi: {label}")

