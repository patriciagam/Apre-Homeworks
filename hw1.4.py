from sklearn.feature_selection import f_classif
from sklearn import tree
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
import numpy as np

# Reading the ARFF file
data = loadarff("./data/diabetes.arff")
df = pd.DataFrame(data[0])

# No need to decode if the outcome is already a string, you can force conversion
df["Outcome"] = df["Outcome"].astype(str)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Feature importance with ANOVA F-value
fimportance = f_classif(X, y)

# Decision Tree Classifier
predictor = tree.DecisionTreeClassifier(min_samples_leaf=20, random_state=1)
predictor.fit(X, y)

# Define class names manually
class_names = ["Positive", "Negative"]  # Adjust based on your dataset

# Plotting the decision tree
figure = plt.figure(figsize=(12, 6))
tree.plot_tree(predictor, feature_names=X.columns.values.tolist(), class_names=class_names)
plt.show()
