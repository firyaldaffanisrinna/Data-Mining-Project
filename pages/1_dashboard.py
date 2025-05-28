import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
found = importlib.util.find_spec("matplotlib")
st.write("matplotlib ditemukan:", found is not None)
st.title("📊 Dashboard Dataset Tumor Otak")
#url
csv_url = "https://raw.githubusercontent.com/firyaldaffanisrinna/Data-Mining-Project/refs/heads/main/avocado_ripeness_dataset.csv"
st.title("Dashboard avocado ripness")
# Load data
def load_data():
df = pd.read_csv(csv_url)
return df
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
