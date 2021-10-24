import pandas as pd

df = pd.read_csv("train.csv")

print(df)

pd.options.plotting.backend = "plotly"

printme = df.groupby(["Fare", "Survived"], as_index=False).size()



fig = printme.hist(x="Fare", y="size", color="Survived")



print(printme)
print(df.info())


fig.show()




