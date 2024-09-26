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
plt.title(f'Class-Conditional Density for Best Feature: {best_feature}')
plt.xlabel(best_feature)
plt.ylabel('Density')

# Density plot for the worst discriminative feature
plt.subplot(1, 2, 2)
sns.kdeplot(data=df, x=worst_feature, hue="Outcome", fill=True, common_norm=False)
plt.title(f'Class-Conditional Density for Worst Feature: {worst_feature}')
plt.xlabel(worst_feature)
plt.ylabel('Density')

# Show the plots
plt.tight_layout()
plt.show()

# # Convert the target column (assumed to be named "Outcome") from bytes to strings
# df["Outcome"] = df["Outcome"].apply(lambda x: x.decode('utf-8'))

# # Split into features (X) and target (y)
# X = df.drop(columns=["Outcome"])  # 8 biological features
# y = df["Outcome"].apply(lambda x: 1 if x == 'diabetes' else 0)  # Convert 'normal' to 0 and 'diabetes' to 1

# # Perform ANOVA test (f_classif) to get F-values and p-values
# F_values, p_values = f_classif(X, y)

# # Identify the best and worst discriminative features
# best_var_idx = np.argmax(F_values)  # Best variable (highest F)
# worst_var_idx = np.argmin(F_values)  # Worst variable (lowest F)

# best_var = X.columns[best_var_idx]
# worst_var = X.columns[worst_var_idx]

# print(f"Best variable: {best_var} (F = {F_values[best_var_idx]})")
# print(f"Worst variable: {worst_var} (F = {F_values[worst_var_idx]})")

# # Plot the class-conditional probability density functions using histograms
# plt.figure(figsize=(14, 6))

# # Histogram for the best variable
# plt.subplot(1, 2, 1)
# for class_value in [0, 1]:  # 0: Normal, 1: Diabetes
#     plt.hist(X[best_var][y == class_value], bins=15, alpha=0.5, label=f'Class {class_value}')
# plt.title(f'Best Variable: {best_var}')
# plt.xlabel(f'{best_var}')
# plt.ylabel('Frequency')
# plt.legend()

# # Histogram for the worst variable
# plt.subplot(1, 2, 2)
# for class_value in [0, 1]:
#     plt.hist(X[worst_var][y == class_value], bins=15, alpha=0.5, label=f'Class {class_value}')
# plt.title(f'Worst Variable: {worst_var}')
# plt.xlabel(f'{worst_var}')
# plt.ylabel('Frequency')
# plt.legend()

# plt.tight_layout()
# plt.show()

