import pandas as pd

df = pd.read_csv("train.csv")

print(df)

pd.options.plotting.backend = "plotly"

df.loc[df["Cabin"].isna(),"Cabin"] = "N"
df["CabinType"] = df["Cabin"].str[0]

printme = df.groupby(["CabinType", "Survived"], as_index=False).size().sort_values(by="CabinType")



fig = printme.hist(x="CabinType", y="size", color="Survived")



print(printme)
print(df.info())


fig.show()




