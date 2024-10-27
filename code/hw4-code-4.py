df = pd.read_csv("./data/accounts.csv")

X = df.iloc[:, :8]   

X.dropna(inplace = True)
X.drop_duplicates(inplace = True)

X["cluster"] = clusters

sns.displot(
    data = X, 
    y = "job",
    hue = "cluster",
    multiple = "dodge",
    stat = "density",
    shrink = 0.8,
    common_norm = False,
    palette = "Set2"
)

sns.displot(
    data = X,  
    y = "education",
    hue = "cluster",
    multiple="dodge",
    stat = "density",
    shrink = 0.8,
    common_norm = False,
    palette = "Set2"
)

plt.show()