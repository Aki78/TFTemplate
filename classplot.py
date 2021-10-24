import pandas as pd


df = pd.read_csv("train.csv")

print(df)

pd.options.plotting.backend = "plotly"

printme = df.groupby(["Pclass", "Survived"], as_index=False).size()

fig = printme.hist(x="Pclass", y="size", color="Survived")
print(printme)


fig.show()




