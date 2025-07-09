# cancer_detection_ui.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from collections import Counter

st.set_page_config(page_title="Cancer Detection using SMOTE", layout="centered")
st.title("🧬 Breast Cancer Detection (SMOTE + Random Forest)")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
y = 1 - y  # Flip labels to make malignant (1) minority

st.write("\n### 🔍 Original Class Distribution:")
st.code(str(Counter(y)), language='python')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

st.write("### ✅ After SMOTE Class Distribution:")
st.code(str(Counter(y_resampled)), language='python')

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
st.write("### 📋 Model Performance")

st.subheader("Confusion Matrix")
st.text(confusion_matrix(y_test, y_pred))

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("ROC AUC Score")
st.success(f"ROC AUC: {round(roc_auc_score(y_test, y_proba), 4)}")

# User Prediction
st.write("---")
st.header("🔮 Predict Cancer Type")
st.markdown("_Enter feature values manually below:_")

input_data = []
for feature in data.feature_names:
    val = st.number_input(label=feature, value=float(X[feature].mean()), format="%0.3f")
    input_data.append(val)

if st.button("Predict"):
    user_input = np.array(input_data).reshape(1, -1)
    pred = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0][1]

    if pred == 1:
        st.error(f"🔴 Prediction: Malignant (Cancer) | Confidence: {proba:.2%}")
    else:
        st.success(f"🟢 Prediction: Benign (No Cancer) | Confidence: {1-proba:.2%}")
