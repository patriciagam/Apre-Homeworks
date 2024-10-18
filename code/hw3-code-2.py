import matplotlib.cm as cm
import matplotlib.colors as mcolors

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

mlp = MLPRegressor(hidden_layer_sizes = (10, 10), random_state = 0)

param_grid = {
    "alpha": [0.0001, 0.001, 0.01],  
    "learning_rate_init": [0.001, 0.01, 0.1],  
    "batch_size": [32, 64, 128]  
}

grid_search = GridSearchCV(mlp, param_grid, scoring = "neg_mean_absolute_error")
grid_search.fit(X_train, y_train)

results = grid_search.cv_results_

alphas = results["param_alpha"].data
learning_rates = results["param_learning_rate_init"].data
batch_sizes = results["param_batch_size"].data
mean_absolute_errors = -results["mean_test_score"]

print("Best parameters found: ", grid_search.best_params_)

fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(111, projection = "3d")

sc = ax.scatter(alphas, batch_sizes, learning_rates, c = mean_absolute_errors, cmap = cm.jet, s = 100)

ax.set_xlabel("L2 penalty")
ax.set_ylabel("Batch Size")
ax.set_zlabel("Learning Rate")

cbar = plt.colorbar(sc, ax=ax, label = "Mean Absolute Error")

plt.show()