import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.title("🔍 Prediksi Kematangan Avocado")

# Load dataset
df = pd.read_csv("avocado_ripeness_dataset.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["ripeness"])
y = df["ripeness"]

# Encode data kategorikal jika ada
X = pd.get_dummies(X)

# Latih model (Decision Tree sederhana)
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Input manual dari user
st.subheader("📝 Masukkan Data Baru")
user_input = {}
for col in X.columns:
    dtype = df[col].dtype if col in df.columns else "float"
    if dtype == "object":
        user_input[col] = st.selectbox(f"{col}:", sorted(df[col].unique()))
    else:
        user_input[col] = st.number_input(f"{col}:", value=float(df[col].mean()))

# Ubah input user menjadi DataFrame
input_df = pd.DataFrame([user_input])

# Sesuaikan kolom dummy (kalau ada)
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Tombol prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    st.success(f"🍃 Prediksi tingkat kematangan: **{prediction}**")
