
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score
)

# Load dataset
df = pd.read_csv("Pancreatic_Cancer.csv")

# Remove rows with missing values
df = df.dropna().copy()

# Target: predict the diagnosed cancer stage
target = "Stage_at_Diagnosis"

# Exclude identifiers and information that may be known only
# after diagnosis or treatment
excluded_columns = [
    "Country",
    "Stage_at_Diagnosis",
    "Survival_Time_Months",
    "Survival_Status",
    "Treatment_Type"
]

X = df.drop(columns=excluded_columns)
y = df[target]

# Identify categorical and numerical features
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Encode categorical features; preserve numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build a consistent preprocessing and prediction pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone

# Test two models using the same training and test data
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

best_accuracy = -1
best_model = None
best_name = ""

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier)
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=pipeline.classes_
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=pipeline.classes_
    )

    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.show()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_name = name

# Save the better-performing model
joblib.dump(best_model, "pancreatic_stage_model.joblib")

print(f"\nBest model: {best_name}")
print(f"Best accuracy: {best_accuracy:.4f}")
print("Saved model: pancreatic_stage_model.joblib")

print("\nStage distribution in the dataset:")
print(y.value_counts())

print("\nAverage age for each stage:")
print(df.groupby("Stage_at_Diagnosis")["Age"].mean().round(2))

print("\nAverage weight loss for each stage:")
print(df.groupby("Stage_at_Diagnosis")["Weight_Loss"].mean().round(2))

print("\nUnique values in selected columns:")

columns_to_check = [
    "Age",
    "Weight_Loss",
    "Smoking_History",
    "Alcohol_Consumption",
    "Stage_at_Diagnosis"
]

for column in columns_to_check:
    print(f"\n{column}:")
    print("Number of unique values:", df[column].nunique())
    print(df[column].value_counts().head(10))

print("\nFeature rates by cancer stage:")

features_to_check = [
    "Smoking_History",
    "Obesity",
    "Diabetes",
    "Chronic_Pancreatitis",
    "Family_History",
    "Jaundice",
    "Weight_Loss"
]

for feature in features_to_check:
    print(f"\n{feature} (% with value 1):")
    rates = df.groupby("Stage_at_Diagnosis")[feature].mean() * 100
    print(rates.round(2))

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

print("\nBaseline comparison:")

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

print(
    "Baseline accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)
print(
    "Baseline balanced accuracy:",
    round(balanced_accuracy_score(y_test, baseline_predictions), 4)
)

print(
    "Random Forest balanced accuracy:",
    round(
        balanced_accuracy_score(
            y_test,
            best_model.predict(X_test)
        ),
        4
    )
)

print("\nDataset quality checks:")

print("Total rows:", len(df))
print("Duplicate rows:", df.duplicated().sum())


print("\nUnique records per stage:")

features_only = df.drop(columns=["Stage_at_Diagnosis"])

for stage in df["Stage_at_Diagnosis"].unique():
    stage_rows = features_only[
        df["Stage_at_Diagnosis"] == stage
    ]

    print(
        stage,
        ":",
        stage_rows.drop_duplicates().shape[0],
        "unique feature records"
    )

print("\nStage distribution (%):")
print(
    (df["Stage_at_Diagnosis"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nStage relationship with all input features:")

feature_columns = [
    col for col in df.columns
    if col not in [
        "Stage_at_Diagnosis",
        "Survival_Time_Months",
        "Survival_Status",
        "Treatment_Type",
        "Country"
    ]
]

for feature in feature_columns:
    if df[feature].nunique() <= 10:
        print(f"\n{feature}:")
        print(pd.crosstab(
            df[feature],
            df["Stage_at_Diagnosis"],
            normalize="columns"
        ).round(2))
