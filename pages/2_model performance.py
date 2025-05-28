import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("📉 Model Performance")

# Load dataset
df = pd.read_csv("avocado_ripeness_dataset.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["ripeness"])
y = df["ripeness"]

# Tangani data kategorikal
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Tampilkan metrik
st.subheader("🎯 Akurasi Model")
st.metric(label="Akurasi", value=f"{accuracy:.2%}")

st.subheader("📊 Confusion Matrix")
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)
ax.set_xticks(range(len(model.classes_)))
ax.set_yticks(range(len(model.classes_)))
ax.set_xticklabels(model.classes_)
ax.set_yticklabels(model.classes_)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
st.pyplot(fig)

st.subheader("📋 Classification Report")
st.dataframe(pd.DataFrame(report).transpose())
