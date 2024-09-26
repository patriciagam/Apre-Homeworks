from sklearn.feature_selection import f_classif
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
import numpy as np

data = loadarff("./data/diabetes.arff")
df = pd.DataFrame(data[0])
df["Outcome"] = df["Outcome"].str.decode("utf-8")

X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

fimportance = f_classif(X, y)

max_f_index = fimportance[0].argmax()  
min_f_index = fimportance[0].argmin()  

print(f"Best discriminative feature: {X.columns.values[max_f_index]} (F = {fimportance[0][max_f_index]})")
print(f"Worst discriminative feature: {X.columns.values[min_f_index]} (F = {fimportance[0][min_f_index]})")

best_feature = X.columns.values[max_f_index]
worst_feature = X.columns.values[min_f_index]

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.kdeplot(
    data = df,
    x = best_feature,
    hue = "Outcome",
    fill = True,
    common_norm = False
)
plt.title("Highest discriminative power")
plt.xlabel(best_feature)
plt.ylabel("Probability")
plt.legend(("Normal", "Diabetes"))

plt.subplot(1, 2, 2)
sns.kdeplot(
    data = df,
    x = worst_feature,
    hue = "Outcome",
    fill = True,
    common_norm = False
)
plt.title("Lowest discriminative power")
plt.xlabel(worst_feature)
plt.ylabel("Probability")
plt.legend(("Normal", "Diabetes"))

plt.show()
