import pandas as pd

df = pd.read_csv("./results/summary.csv")

for index, row in df.iterrows():
    for col in df.columns:
        print(f"{col}: {row[col]}")
    print()