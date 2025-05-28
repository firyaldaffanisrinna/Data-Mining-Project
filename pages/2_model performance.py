import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("📉 Model Performance")

# Load dataset
df = pd.read_csv("avocado_ripeness_dataset.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["ripeness"])
y = df["ripeness"]

# Tangani data kategorikal jika ada (e.g., color_category)
X = pd.get_dummies(X)

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Hitung metrik
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Tampilkan hasil
st.subheader("🎯 Akurasi Model")
st.metric(label="Akurasi", value=f"{accuracy:.2%}")

st.subheader("📊 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
st.pyplot(fig)

st.subheader("📋 Classification Report")
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)
