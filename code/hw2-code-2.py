from sklearn.preprocessing import MinMaxScaler
import numpy as np

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

min_max_scaler = MinMaxScaler()
X_scaled_minmax = X.copy()
X_scaled_minmax[numeric_features] = min_max_scaler.fit_transform(X[numeric_features])

knn_scores_minmax = cross_val_score(knn_classifier, X_scaled_minmax, y, cv = skf, scoring = "accuracy")
gnb_scores_minmax = cross_val_score(naive_bayes_classifier, X_scaled_minmax, y, cv = skf, scoring = "accuracy")

print(f"Naïve Bayes accuracy = {np.mean(gnb_scores_minmax):.2f}, {np.std(gnb_scores_minmax):.2f}")
print(f"kNN accuracy = {np.mean(knn_scores_minmax):.2f}, {np.std(knn_scores_minmax):.2f}")