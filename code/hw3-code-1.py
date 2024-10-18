from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

df = pd.read_csv("./data/parkinsons.csv")

X = df.drop("target", axis = 1)
y = df["target"]

warnings.filterwarnings(action = "ignore", category = ConvergenceWarning)

random_seeds = range(1, 11)

results = {"Linear Regression": [], "MLP No Activation": [], "MLP ReLU": []}

for i in random_seeds:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = i)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["Linear Regression"].append(mean_absolute_error(y_test, y_pred_lr))

    mlp_no_act = MLPRegressor(hidden_layer_sizes = (10, 10), activation ="identity", random_state = 0)
    mlp_no_act.fit(X_train, y_train)
    y_pred_mlp_no_act = mlp_no_act.predict(X_test)
    results["MLP No Activation"].append(mean_absolute_error(y_test, y_pred_mlp_no_act))

    mlp_relu = MLPRegressor(hidden_layer_sizes = (10, 10), activation = "relu", random_state = 0)
    mlp_relu.fit(X_train, y_train)
    y_pred_mlp_relu = mlp_relu.predict(X_test)
    results["MLP ReLU"].append(mean_absolute_error(y_test, y_pred_mlp_relu))

results_df = pd.DataFrame(results)

plt.figure(figsize = (10, 10))

sns.boxplot(data = results_df, palette = "Set2", width = 0.5)
plt.ylabel('Mean Absolute Error')

plt.show()