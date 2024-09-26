from sklearn.feature_selection import f_classif
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
import numpy as np

# Reading the ARFF file
data = loadarff("./data/diabetes.arff")
df = pd.DataFrame(data[0])
df["Outcome"] = df["Outcome"].str.decode("utf-8")

X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

fimportance = f_classif(X, y)

from sklearn.feature_selection import f_classif

# Perform ANOVA F-test on the dataset
fimportance = f_classif(X, y)

# Find the index of the max and min F-scores
max_f_index = fimportance[0].argmax()  
min_f_index = fimportance[0].argmin()  

# Print the best and worst discriminative features
print(f"Best discriminative feature: {X.columns.values[max_f_index]} (F = {fimportance[0][max_f_index]})")
print(f"Worst discriminative feature: {X.columns.values[min_f_index]} (F = {fimportance[0][min_f_index]})")

best_feature = X.columns.values[max_f_index]
worst_feature = X.columns.values[min_f_index]

# Plot the class-conditional probability density functions for the best feature
plt.figure(figsize=(14, 6))

# Density plot for the best discriminative feature
plt.subplot(1, 2, 1)

sns.kdeplot(data=df, x=best_feature, hue="Outcome", fill=True, common_norm=False)
plt.title(f"Highest discriminative power")
plt.legend(("Normal", "Diabetes"))
plt.xlabel(best_feature)
plt.ylabel("Probability")

# Density plot for the worst discriminative feature
plt.subplot(1, 2, 2)
sns.kdeplot(data=df, x=worst_feature, hue="Outcome", fill=True, common_norm=False)
plt.title(f"Lowest discriminative power")
plt.legend(("Normal", "Diabetes"))
plt.xlabel(worst_feature)
plt.ylabel("Probability")

# Show the plots
plt.tight_layout()
plt.show()