
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Debernardi et al 2020 data.csv")

# Create binary labels:
# Diagnosis 1 and 2 = No Cancer
# Diagnosis 3 = Cancer
df["Cancer_Status"] = df["diagnosis"].map({
    1: "No Cancer",
    2: "No Cancer",
    3: "Cancer"
})

# Select useful features
features = [
    "age",
    "sex",
    "plasma_CA19_9",
    "creatinine",
    "LYVE1",
    "REG1B",
    "TFF1",
    "REG1A"
]

X = df[features].copy()
y = df["Cancer_Status"]

# Convert sex into numeric form
X["sex"] = X["sex"].map({"M": 1, "F": 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build the model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "pancreatic_detection_model.joblib")

print("\nModel saved as pancreatic_detection_model.joblib")