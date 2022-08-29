import pandas as pd

df = pd.read_csv("data/qlik_aug.csv", low_memory=False)

df= df[df["visit_date_range"] == "Baseline"]

print(df["food_did_not_last"].value_counts())
print(df["buying_food"].value_counts())
