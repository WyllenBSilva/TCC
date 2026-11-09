import pandas as pd

df = pd.read_parquet("dados_parquet/RDSC2401.parquet")

print(df.columns.tolist())