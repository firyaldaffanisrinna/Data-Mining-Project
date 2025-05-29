import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("🥑 Prediksi Kematangan Avocado")

# Load dataset
df = pd.read_csv("avocado_ripeness_dataset.csv")

# Tentukan target dan fitur
target_col = "ripeness"
X = df.drop(columns=[target_col])
y = df[target_col]

# Simpan info kolom numerik & kategorikal
numerik_cols = X.select_dtypes(include="number").columns.tolist()
kategori_cols = X.select_dtypes(include="object").columns.tolist()

# One-hot encoding untuk kolom kategorikal
X_encoded = pd.get_dummies(X)
input_columns = X_encoded.columns

# Train model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_encoded, y)
train_accuracy = model.score(X_encoded, y)

# Tampilkan akurasi training
st.markdown(f"📈 Akurasi model pada data latih: `{train_accuracy*100:.2f}%`")

# Input user
st.subheader("📝 Masukkan Data Avocado Baru")
user_input = {}

# Input numerik
for col in numerik_cols:
    max_val = float(X[col].max())
    val = st.number_input(f"{col}", min_value=0.0, max_value=max_val, value=0.0)
    user_input[col] = val

# Input kategorikal
for col in kategori_cols:
    pilihan = st.selectbox(f"{col}", sorted(X[col].unique()))
    user_input[col] = pilihan

# Preprocessing input user
user_df = pd.DataFrame([user_input])
user_encoded = pd.get_dummies(user_df)

# Pastikan kolom sesuai dengan data training
user_encoded = user_encoded.reindex(columns=input_columns, fill_value=0)

# Prediksi
if st.button("🔮 Prediksi"):
    pred = model.predict(user_encoded)[0]
    st.success(f"🍃 Prediksi tingkat kematangan: **{pred}**")
