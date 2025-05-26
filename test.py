import pandas as pd

df1 = pd.read_csv(r"C:\Users\39370\Downloads\verifica_transazioni.csv")
df2 = pd.read_csv(r"C:\Users\39370\Downloads\verifica_transazioni2.csv")
print(df1)
print(df2)
val1 = df1["date"].tolist()
val2 = df2["date"].tolist()
print(len(list(set(val1))), len(list(set(val2))))
vals = set(val1) - set(val2)
print(vals)

for i in range(len(df1)):
    if df1["date"][i] != df2["date"][i]:
        print(df1.iloc[[i]])
        print(df2.iloc[[i]])
        print("==============================")
