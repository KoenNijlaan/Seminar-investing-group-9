from pathlib import Path
import pandas as pd

data_path = Path("data_intermediate/converted_parquet")
file_path = sorted(data_path.glob("*.parquet"))[0]

print(f"Reading: {file_path.name}")

df = pd.read_parquet(file_path)

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())

r = df["returns_5m"].iloc[0]
print("\nreturns_5m first entry:")
print(type(r))
print("Length:", len(r))
print("First 5 values:", r[:5])