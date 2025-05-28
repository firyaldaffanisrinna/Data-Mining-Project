import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
streamlit run pages/1_dashboard.py

import importlib.util
found = importlib.util.find_spec("matplotlib")
st.write("matplotlib ditemukan:", found is not None)
st.title("📊 Dashboard Dataset Tumor Otak")

# Load data
df = pd.read_csv("dataset.csv")

# Tabel data
st.subheader("📋 Tabel Data")
st.dataframe(df)

# Statistik ringkas
st.subheader("📈 Statistik Ringkas")
st.write(df.describe(include='all'))

# Visualisasi
st.subheader("📊 Visualisasi Jumlah File per Label")
fig, ax = plt.subplots()
ax.bar(df['label'], df['number_of_files'], color='mediumseagreen')
ax.set_xlabel("Jenis Tumor")
ax.set_ylabel("Jumlah File")
ax.set_title("Distribusi Data Gambar")
st.pyplot(fig)
