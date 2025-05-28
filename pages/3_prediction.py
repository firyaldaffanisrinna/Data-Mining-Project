import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.title("🔍 Prediksi Kematangan Avocado")

# Load dataset
df = pd.read_csv("avocado_ripeness_dataset.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["ripeness"])
y = df["ripeness"]

# Simpan nilai maksimum kolom numerik sebelum encoding
max_numerik = X.select_dtypes(include='number').max()

# One-hot encoding
X_encoded = pd.get_dummies(X)

# Latih model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_encoded, y)

# Input manual dari user
st.subheader("📝 Masukkan Data Baru")
user_input = {}

for col in X_encoded.columns:
    if X_encoded[col].dtype == 'uint8':
        # Dummy kategorikal
        label = col.split('_')[-1]
        pilihan = st.selectbox(f"{col} (Pilih Ya/Tidak):", ["Tidak", "Ya"])
        user_input[col] = 1 if pilihan == "Ya" else 0
    else:
        # Kolom numerik
        asal_nama = col.split("_")[0]  # Ambil nama aslinya
        max_val = max_numerik.get(asal_nama, 100.0)  # Default aman jika tak ditemukan
        user_input[col] = st.number_input(f"{col}:", min_value=0.0, max_value=float(max_val), value=0.0)

# Konversi ke DataFrame
input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=X_encoded.columns, fill_value=0)

# Tombol prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    st.success(f"🍃 Prediksi tingkat kematangan: **{prediction}**")
