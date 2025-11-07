import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
# --- Load the data (Ensure 'Crop.csv' is uploaded to your Colab session files) ---
file_path = 'Crop.csv'
df = pd.read_csv(file_path)

# --- Define Target (y) and Features (X) ---
target_column = 'label'
X = df.drop(target_column, axis=1)
y = df[target_column]

# --- Data Split (80% Train, 20% Test) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define all features as numerical (since they are N, P, K, etc.)
numerical_features = X.columns.tolist()
# 1. Define the Preprocessing Step (StandardScaler is used since all features are numerical)
scaler = StandardScaler()

# 2. Model 1: Logistic Regression Pipeline (A linear baseline model)
# The pipeline scales the data and then trains the classifier
model_lr_pipeline = Pipeline(steps=[('scaler', scaler),
                                    ('classifier', LogisticRegression(multi_class='auto', solver='liblinear', random_state=42))])

# 3. Model 2: Random Forest Classifier Pipeline (A powerful ensemble model)
model_rf_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# --- Train Models ---
print("Training Logistic Regression...")
model_lr_pipeline.fit(X_train, y_train)

print("Training Random Forest Classifier...")
model_rf_pipeline.fit(X_train, y_train)

# Make predictions for evaluation
y_pred_lr = model_lr_pipeline.predict(X_test)
y_pred_rf = model_rf_pipeline.predict(X_test)
# Calculate overall accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\n" + "="*50)
print("             STAGE 5: MODEL EVALUATION RESULTS")
print("="*50)

# 1. Model Comparison Summary Table
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Test Accuracy': [accuracy_lr, accuracy_rf]
})
print("\n--- Model Comparison Summary ---")
print(results.sort_values(by='Test Accuracy', ascending=False).to_markdown(index=False))

# 2. Detailed Classification Report for the best model (Random Forest)
print("\n--- Detailed Classification Report (Random Forest) ---")
print(classification_report(y_test, y_pred_rf))
# 1. Confusion Matrix Plot (Shows prediction vs. actual crop)
print("\nGenerating Confusion Matrix...")
fig, ax = plt.subplots(figsize=(15, 15))
ConfusionMatrixDisplay.from_estimator(model_rf_pipeline, X_test, y_test, ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
ax.set_title("Confusion Matrix for Crop Recommendation (Random Forest)")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved Confusion Matrix as 'confusion_matrix.png'")
plt.close()

# 2. Feature Importance Plot (Shows most influential factors)
print("Generating Feature Importance Chart...")
importances = model_rf_pipeline.named_steps['classifier'].feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importance for Crop Recommendation')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved Feature Importance Chart as 'feature_importance.png'")
plt.close()

print("\n" + "="*50)
print("PROJECT COMPLETE. Check your Colab files for the two image plots!")
print("="*50)


import joblib  # joblib is more efficient for large numpy objects like sklearn models

# Save both trained models
joblib.dump(model_lr_pipeline, 'model_logistic_regression.pkl')
joblib.dump(model_rf_pipeline, 'model_random_forest.pkl')

print("\n" + "="*50)
print("STAGE 7: TRAINED MODELS SAVED SUCCESSFULLY")
print("="*50)
print("Saved files:")
print(" - model_logistic_regression.pkl")
print(" - model_random_forest.pkl")
print("\nYou can load them later with:")
print("   model = joblib.load('model_random_forest.pkl')")
