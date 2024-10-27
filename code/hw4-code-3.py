import seaborn as sns

kmeans = KMeans(n_clusters = 3, random_state = 42)
clusters = kmeans.fit_predict(X_normalized)  

pca = PCA(n_components = 2)
pca_components = pca.fit_transform(X_normalized)

selected_features = list(X_normalized.var().sort_values(ascending = False).head(2).index)


plt.figure(figsize = (8, 6))
sns.scatterplot(x = pca_components[:, 0], y = pca_components[:, 1], hue = X["cluster"], palette = "Set2")
plt.xlabel(selected_features[0] + " (component 1)")
plt.ylabel(selected_features[1] + " (component 2)")
plt.title("Scatterplot of Clusters According to First 2 Principal Components")

plt.show()