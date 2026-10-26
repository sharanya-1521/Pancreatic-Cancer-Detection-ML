
import pandas as pd

df = pd.read_csv("Debernardi et al 2020 data.csv")

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

for col in df.columns:
    if any(word in col.lower() for word in
           ["cancer", "diagnosis", "stage", "status", "target"]):
        print(f"\nValues in {col}:")
        print(df[col].value_counts(dropna=False).head(20))
