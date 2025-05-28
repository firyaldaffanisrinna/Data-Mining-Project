import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
found = importlib.util.find_spec("matplotlib")
st.write("matplotlib ditemukan:", found is not None)
st.title("📊 Dashboard Dataset Tumor Otak")

# Load data
df = pd.read_csv("avocado_ripeness_dataset.csv")
st.write("Kolom tersedia:", df.columns.tolist())

# Tabel data
st.subheader("📋 Tabel Data")
st.dataframe(df)

# Statistik ringkas
st.subheader("📈 Statistik Ringkas")
st.write(df.describe(include='all'))

# Visualisasi
st.subheader("📊 Visualisasi Jumlah Data per Kematangan (Ripeness)")
ripeness_counts = df['ripeness'].value_counts()
fig, ax = plt.subplots()
ax.bar(ripeness_counts.index, ripeness_counts.values, color='mediumseagreen')
ax.set_xlabel("Tingkat Kematangan")
ax.set_ylabel("Jumlah Data")
st.pyplot(fig)

