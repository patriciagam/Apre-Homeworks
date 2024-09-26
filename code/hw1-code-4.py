from sklearn.feature_selection import f_classif
from sklearn import tree
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
import numpy as np

data = loadarff("./data/diabetes.arff")
df = pd.DataFrame(data[0])
df["Outcome"] = df["Outcome"].astype(str)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

fimportance = f_classif(X, y)

predictor = tree.DecisionTreeClassifier( max_depth = 3, random_state = 1)
predictor.fit(X, y)

class_names = ["Normal", "Diabetes"]  

figure = plt.figure(figsize = (12, 6))
tree.plot_tree(predictor, feature_names = X.columns.values.tolist(), class_names = class_names)

plt.show()
