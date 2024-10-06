from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("./data/heart-disease.csv")

X = df.drop("target", axis = 1)
y = df["target"]

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

knn_classifier = KNeighborsClassifier(n_neighbors = 5)
naive_bayes_classifier = GaussianNB()

knn_scores = cross_val_score(knn_classifier, X, y.tolist(), cv = skf, scoring='accuracy')
naive_bayes_scores = cross_val_score(naive_bayes_classifier, X, y.tolist(), cv = skf, scoring = 'accuracy')

plt.figure(figsize=(10, 10))

sns.boxplot(data = [knn_scores, naive_bayes_scores], palette = "Set2", width = 0.4)
plt.xticks([0, 1], ["kNN (k=5)", "Naïve Bayes"], fontsize = 11)
plt.ylabel("Accuracy", fontsize = 12)

plt.show()