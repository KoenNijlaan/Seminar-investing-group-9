from pathlib import Path
import pandas as pd

# Pick one RSJ file
file_path = sorted(Path("data_intermediate/rsj_daily").glob("*.parquet"))[0]

print(f"Reading: {file_path.name}")

df = pd.read_parquet(file_path)

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head(10))

print("\nInfo:")
print(df.info())

print("\nRSJ summary:")
print(df["rsj"].describe())