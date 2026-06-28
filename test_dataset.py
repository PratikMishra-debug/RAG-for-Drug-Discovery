import pandas as pd

data = pd.read_csv("data/drug_dataset.csv")

print(data.head())
print(len(data))