import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🧪 Pancreatic Cancer Detection")
st.write("Predict the stage of pancreatic cancer based on patient data.")

# Load dataset
df = pd.read_csv("Pancreatic_Cancer.csv")
df.drop(['Country'], axis=1, inplace=True)
df.dropna(inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df.drop('Stage_at_Diagnosis', axis=1)
y = df['Stage_at_Diagnosis']

# Fit scaler and model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Input UI
st.subheader("Enter patient data:")

user_input = {}
for col in X.columns:
    if col in categorical_cols:
        options = list(label_encoders[col].classes_)
        selected = st.selectbox(f"{col}:", options)
        user_input[col] = label_encoders[col].transform([selected])[0]
    else:
        value = st.number_input(f"{col}:", min_value=0.0, step=0.1)
        user_input[col] = value

# Create input DataFrame
input_df = pd.DataFrame([user_input])

# Predict on button click
if st.button("Predict"):
    input_df_scaled = scaler.transform(input_df)  # Scale the user input
    prediction = rf_model.predict(input_df_scaled)[0]

    st.subheader(f"🎯 Predicted Stage at Diagnosis: {prediction}")

    stage_info = {
        0: "Stage 0 - Abnormal cells, not yet cancer.",
        1: "Stage I - Small tumor, localized.",
        2: "Stage II - Larger tumor, may spread to nearby tissue.",
        3: "Stage III - Advanced, spread to major blood vessels or distant tissues."
    }
    st.info(stage_info.get(prediction, "Unknown stage"))



