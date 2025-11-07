# 🌾 Crop Recommendation System: A Machine Learning Approach

## 🧠 Overview
This project designs and implements a **Multi-Class Classification** model to recommend the most suitable crop for cultivation based on soil and environmental parameters.

It acts as a **decision-support system for precision agriculture**, guiding farmers to make data-driven decisions that optimize land use and maximize productivity.

The project includes a **comparative analysis** between two machine learning algorithms:
- **Logistic Regression**
- **Random Forest Classifier**

---

## 🎯 Project Goal
To accurately predict the **best crop to grow** based on the following seven input features:

| Feature | Description |
|----------|-------------|
| **N** | Ratio of Nitrogen content in the soil |
| **P** | Ratio of Phosphorus content in the soil |
| **K** | Ratio of Potassium content in the soil |
| **temperature** | Temperature in °C |
| **humidity** | Relative humidity (%) |
| **ph** | pH value of the soil |
| **rainfall** | Rainfall in mm |

---

## ⚙️ Methodology

### 1. Data Preprocessing
- All numerical features were **standardized using `StandardScaler`** to ensure equal contribution from variables with different ranges (e.g., pH vs. rainfall).

### 2. Model Training
Two models were trained on the preprocessed data:
- **Logistic Regression:** Used as a baseline linear model.  
- **Random Forest Classifier:** An ensemble model that captures complex, non-linear relationships.

### 3. Model Persistence
The final trained model pipelines were **saved using `joblib`** for deployment and future use.

---

## ✅ Key Results & Performance

### 1. Model Comparison

| Model | Test Accuracy |
|--------|----------------|
| **Random Forest** | 🟢 0.9932 (99.32%) |
| **Logistic Regression** | 🟡 0.9295 (92.95%) |

### 2. Feature Importance
Analysis revealed that **Potassium (K)** and **Phosphorus (P)** are the most influential factors for accurate crop prediction.

> 💡 *Insight:* Soil composition (nutrients) plays a more dominant role than ambient conditions (like temperature or rainfall) in this dataset.

### 3. Confusion Matrix
The confusion matrix confirms high reliability — predictions are heavily concentrated along the diagonal, indicating strong classification performance.

---

## 📂 Repository Structure

📦 Crop-Recommendation-System

┣ 📄 Crop.csv # Dataset used for training/testing

┣ 📓 Crop_Recommendation.ipynb # Main Colab notebook (code & analysis)

┣ 📦 random_forest_model.joblib # Saved Random Forest pipeline

┣ 📦 logistic_regression_model.joblib # Saved Logistic Regression pipeline

┣ 📊 confusion_matrix.png # Confusion matrix visualization

┣ 📈 feature_importance.png # Feature importance plot

┗ 📜 README.md # Project documentation
## 📊 Future Improvements
Integrate Deep Learning models (e.g., ANN) for comparison

Add web interface using Flask or Streamlit for real-time crop recommendation

Extend dataset to include regional soil types and seasonal variations

---

## 💻 Dependencies

Install the following Python libraries before running the notebook:

bash:
pip install pandas numpy scikit-learn matplotlib seaborn joblib

---
## 🧩 Author
Telugu Rajesh Kumar

AI & ML Enthusiast | Aspiring Software Engineer

📎 [github.com/rajeshkumarhh]
___
