import pandas as pd

df = pd.read_csv("comments.csv")

for x in list(df["body"]):
    print(x)