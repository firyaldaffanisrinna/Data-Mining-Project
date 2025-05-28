import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Dataset Overview")

# Load data
df = pd.read_csv("dataset.csv")

st.write("### Tabel Data")
st.dataframe(df)

# Visualisasi
st.write("### Visualisasi Jumlah File")
fig, ax = plt.subplots()
ax.bar(df['label'], df['number_of_files'], color='teal')
ax.set_ylabel("Jumlah File")
ax.set_xlabel("Label")
ax.set_title("Jumlah File per Label Tumor")
st.pyplot(fig)

