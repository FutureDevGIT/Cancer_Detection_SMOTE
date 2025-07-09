# 🧬 Breast Cancer Detection using SMOTE + Random Forest

A simple yet powerful ML project that handles **imbalanced datasets** using **SMOTE** and trains a **Random Forest Classifier** to detect cancer from real medical data.  
Comes with an intuitive **Streamlit UI** for live predictions.

---

## 📊 Dataset

- Used: `Breast Cancer Wisconsin` dataset from `sklearn.datasets`
- Original class:  
  - `0`: Malignant (cancer)  
  - `1`: Benign (non-cancer)  
- For imbalance simulation: We flipped the labels → making `1` (Malignant) the **minority** class.

---

## ⚙️ Features

- 🧪 SMOTE oversampling to balance classes  
- 🌲 Random Forest training  
- 📉 Evaluation using: accuracy, precision, recall, F1-score, ROC AUC  
- 🔮 Live prediction via Streamlit UI with user inputs  
- 📊 Confusion Matrix + Classification Report on test set

---

## 🚀 Quick Start

### ✅ Install requirements

```bash
pip install -r requirements.txt
# or manually
pip install streamlit scikit-learn imbalanced-learn pandas numpy
```

### ▶️ Run Streamlit App

```bash
streamlit run cancer_detection_ui.py
```

### 🖥️ Project Structure

```bash
.
├── cancer_detection_smote.py      # Core ML logic and evaluation
├── cancer_detection_ui.py         # Streamlit-based UI for prediction
├── README.md
└── requirements.txt               # (Optional) dependencies
```

### 📈 Model Metrics (Example Output)

```bash
Confusion Matrix:
[[104   3]
 [  5  59]]

Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       107
           1       0.95      0.92      0.93        64

    accuracy                           0.95       171
   macro avg       0.95      0.94      0.94       171
weighted avg       0.95      0.95      0.95       171

ROC AUC Score: 0.9842
```

### 💡 Future Ideas

- 🔁 Add XGBoost/LGBM comparison
- 📤 Export predictions
- ☁️ Deploy to Streamlit Cloud
- 🧬 Add SHAP explainability

### 📜 License
- MIT License © 2025 Mayank Raval.

---

## 📦 Optional: `requirements.txt`

```txt
streamlit
scikit-learn
imbalanced-learn
pandas
numpy
```
