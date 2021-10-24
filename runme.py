import pandas as pd


df = pd.read_csv("train.csv")

print(df)

pd.options.plotting.backend = "plotly"

printme = df.groupby(["Age", "Survived"], as_index=False).size()

printme["ratio"]

fig = printme.hist(x="Age", y="size", color="Survived")
print(printme)


fig.show()




