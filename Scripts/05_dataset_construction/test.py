import pandas as pd
p = pd.read_parquet("data_final/panel/weekly_panel.parquet")
print(p[p["res_total_p025"].notna()][["week", "res_total_p025", "res_sys_p025", "res_idio_p025"]].head())

