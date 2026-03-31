import pandas as pd
p = pd.read_parquet("data_final/panel/weekly_panel.parquet")
print(p["res_weekly"].describe())
print(p["res_weekly"].skew())
