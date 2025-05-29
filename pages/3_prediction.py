import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

st.title("🥑 Prediksi Kematangan Avocado")

# Load dataset
df = pd.read_csv("avocado_ripeness_dataset.csv")

# Pisahkan fitur dan target
target_col = "ripeness"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identifikasi kolom numerik & kategorikal
numerik_cols = X.select_dtypes(include="number").columns.tolist()
kategori_cols = X.select_dtypes(include="object").columns.tolist()

# One-hot encoding
X_encoded = pd.get_dummies(X)
input_columns = X_encoded.columns

# Normalisasi (untuk KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Model tersedia
model_pilihan = st.selectbox("🧠 Pilih Model Klasifikasi", ["Decision Tree", "K-Nearest Neighbors", "Naive Bayes"])

# Latih model
if model_pilihan == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_encoded, y)
    akurasi = model.score(X_encoded, y)
elif model_pilihan == "K-Nearest Neighbors":
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, y)
    akurasi = model.score(X_scaled, y)
elif model_pilihan == "Naive Bayes":
    model = GaussianNB()
    model.fit(X_encoded, y)
    akurasi = model.score(X_encoded, y)

st.markdown(f"📈 Akurasi model pada data latih: `{akurasi*100:.2f}%`")

# Input manual dari user
st.subheader("📝 Masukkan Data Avocado Baru")
user_input = {}

# Input numerik
for col in numerik_cols:
    max_val = float(X[col].max())
    user_input[col] = st.number_input(f"{col}", min_value=0.0, max_value=max_val, value=0.0)

# Input kategorikal
for col in kategori_cols:
    pilihan = st.selectbox(f"{col}", sorted(X[col].unique()), key=col)
    user_input[col] = pilihan

# Proses input
user_df = pd.DataFrame([user_input])
user_encoded = pd.get_dummies(user_df)
user_encoded = user_encoded.reindex(columns=input_columns, fill_value=0)

# Normalisasi input jika model KNN
if model_pilihan == "K-Nearest Neighbors":
    user_encoded_scaled = scaler.transform(user_encoded)
else:
    user_encoded_scaled = user_encoded

if st.button("🔍 Prediksi"):
        hasil = model.predict(user_enc)[0]
        st.success(f"🍃 Prediksi tingkat kematangan: {hasil}")

    if st.button("⬅️ Kembali ke Evaluasi"):
        st.switch_page("pages/2_Model_Performance.py")
# Tombol prediksi
if st.button("🔮 Prediksi"):
    pred = model.predict(user_encoded_scaled)[0]
    st.success(f"🍃 Prediksi tingkat kematangan ({model_pilihan}): **{pred}**")
