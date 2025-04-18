import sys
print("Using Python version:", sys.executable)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Pancreatic_Cancer.csv")

# Drop columns not helpful for prediction
df.drop(['Country'], axis=1, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Encode all categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = df.drop('Stage_at_Diagnosis', axis=1)
y = df['Stage_at_Diagnosis']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Pancreatic Cancer Stage Detection\nUsing Logistic Regression, Decision Tree, and Random Forest\n")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)
print("Logistic Regression\nClassification Report:")
print(classification_report(y_test, log_preds))
print("Accuracy Score:", accuracy_score(y_test, log_preds))

# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)
print("\nDecision Tree\nClassification Report:")
print(classification_report(y_test, tree_preds))
print("Accuracy Score:", accuracy_score(y_test, tree_preds))

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("\nRandom Forest\nClassification Report:")
print(classification_report(y_test, rf_preds))
print("Accuracy Score:", accuracy_score(y_test, rf_preds))

# Feature Importance Plot
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]  # Sort by importance

plt.figure(figsize=(12, 6))
plt.title("Feature Importance (Random Forest)", fontsize=16)
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()









