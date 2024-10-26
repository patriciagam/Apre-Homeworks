from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

numeric_features = ["age", "balance"]

X_normalized = X.copy()
X_normalized[numeric_features] = StandardScaler().fit_transform((X[numeric_features]))

pca = PCA(n_components = 2)
pca.fit(X_normalized)
explained_variance = pca.explained_variance_ratio_

print(explained_variance)    