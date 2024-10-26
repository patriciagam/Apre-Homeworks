import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./data/accounts.csv")

X = df.iloc[:, :8]   

X.dropna(inplace = True)
X.drop_duplicates(inplace = True)
X = pd.get_dummies(X, drop_first = True)

numeric_features = ["age", "balance"]

X_normalized = X.copy()
X_normalized[numeric_features] = MinMaxScaler().fit_transform(X[numeric_features])

k_values = [2, 3, 4, 5, 6, 7, 8]
sse = []

for k in k_values:
    kmeans = KMeans(n_clusters = k, init = "random", random_state = 42, max_iter = 500)
    kmeans_model = kmeans.fit(X_normalized)
    sse.append(kmeans_model.inertia_)

plt.figure(figsize = (10, 8))
plt.plot(k_values, sse, marker = 'o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.grid(True)

plt.show()