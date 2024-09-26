from sklearn import model_selection, tree, metrics
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
import numpy as np

# Reading the ARFF file
data = loadarff("./data/diabetes.arff")
df = pd.DataFrame(data[0])
df["Outcome"] = df["Outcome"].str.decode("utf-8")

# Separate features from the outcome (Outcome)
X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

# List of specific values for min_samples_split to test
MIN_SAMPLES_SPLIT = [2, 5, 10, 20, 30, 50, 100] 
runs = 10  # Number of runs for averaging

# Initialize lists to store average accuracy for each split value
average_training_accuracy = []  # Initialize this list
average_test_accuracy = []  # Initialize this list

for min_samples_split in MIN_SAMPLES_SPLIT:
    train_acc_list = []
    test_acc_list = []

    for _ in range(runs):
        # Split the dataset into a training set (80%) and a testing set (20%)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, train_size = 0.8, stratify = y, random_state = 1 
        )

        # Fit the decision tree classifier
        predictor = tree.DecisionTreeClassifier(min_samples_split = min_samples_split, random_state = 1)
        predictor.fit(X_train, y_train)

        # Use the decision tree to predict the outcomes of the given observations
        y_train_pred = predictor.predict(X_train)
        y_test_pred = predictor.predict(X_test)

        # Get the accuracy of each test
        train_acc = metrics.accuracy_score(y_train, y_train_pred)
        test_acc = metrics.accuracy_score(y_test, y_test_pred)

        # Append accuracies to the respective lists
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    # Average accuracies over the runs
    average_training_accuracy.append(np.mean(train_acc_list))
    average_test_accuracy.append(np.mean(test_acc_list))

# Plot the training and test accuracy for each min_samples_split
plt.plot(
    MIN_SAMPLES_SPLIT,
    average_training_accuracy,  # Use the averaged accuracies for plotting
    label = "Training Accuracy",
    marker="o",
    color="#4caf50",
)
plt.plot(
    MIN_SAMPLES_SPLIT,
    average_test_accuracy,  # Use the averaged accuracies for plotting
    label = "Test Accuracy",
    marker = "x",
    color = "#ff5722",
)

plt.xlabel("Minimum Samples Split")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy vs. Minimum Samples Split")
plt.legend()
plt.grid(True)
plt.savefig("./h1.2-plot.svg")  # Save the plot
plt.show()
