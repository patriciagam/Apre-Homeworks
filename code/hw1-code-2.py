from sklearn import model_selection, tree, metrics
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
import numpy as np

data = loadarff("./data/diabetes.arff")
df = pd.DataFrame(data[0])
df["Outcome"] = df["Outcome"].str.decode("utf-8")

X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

MIN_SAMPLES_SPLIT = [2, 5, 10, 20, 30, 50, 100] 
runs = 10  

X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, train_size = 0.8, stratify = y, random_state = 1 
        )

average_training_accuracy = []  
average_test_accuracy = []  

for min_samples_split in MIN_SAMPLES_SPLIT:
    train_acc_list = []
    test_acc_list = []

    for _ in range(runs):
        predictor = tree.DecisionTreeClassifier(min_samples_split = min_samples_split, random_state = 1)
        predictor.fit(X_train, y_train)
        
        y_train_pred = predictor.predict(X_train)
        y_test_pred = predictor.predict(X_test)

        train_acc = metrics.accuracy_score(y_train, y_train_pred)
        test_acc = metrics.accuracy_score(y_test, y_test_pred)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    average_training_accuracy.append(np.mean(train_acc_list))
    average_test_accuracy.append(np.mean(test_acc_list))

plt.plot(
    MIN_SAMPLES_SPLIT,
    average_training_accuracy,  
    label = "Training Accuracy",
    marker="+",
    color="#4caf50",
)
plt.plot(
    MIN_SAMPLES_SPLIT,
    average_test_accuracy,  
    label = "Test Accuracy",
    marker = ".",
    color = "#ff5722",
)

plt.xlabel("Minimum Samples Split")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy vs. Minimum Samples Split")
plt.legend()
plt.grid(True)

plt.show()