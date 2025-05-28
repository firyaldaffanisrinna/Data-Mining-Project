import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
found = importlib.util.find_spec("matplotlib")
st.write("matplotlib ditemukan:", found is not None)
st.title("📊 Dashboard Dataset Avocado Ripness")

# Load data
df = pd.read_csv("avocado_ripeness_dataset.csv")
st.write("Kolom tersedia:", df.columns.tolist())

# Tabel data
st.subheader("📋 Tabel Data")
st.dataframe(df)

# Statistik ringkas
st.subheader("📈 Statistik Ringkas")
st.write(df.describe(include='all'))

# Hitung jumlah per kategori ripeness
ripeness_counts = df['ripeness'].value_counts()

# Warna khusus untuk setiap kategori (disesuaikan urutannya)
warna_kategori = {
    'ripe': 'red',
    'pre-conditioned': 'yellow',
    'hard': 'green',
    'breaking': 'blue'
}
# Ambil warna sesuai urutan kategori yang muncul
colors = [warna_kategori.get(k, 'gray') for k in ripeness_counts.index]
# Visualisasi
st.subheader("📊 Statistic Jumlah Data per Kematangan (Ripeness)")
fig, ax = plt.subplots()
ax.bar(ripeness_counts.index, ripeness_counts.values, color=color)
ax.set_xlabel("Tingkat Kematangan")
ax.set_ylabel("Jumlah Data")
st.pyplot(fig)

