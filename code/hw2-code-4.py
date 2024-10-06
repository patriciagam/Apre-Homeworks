from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split

K_VALUES = [1, 5, 10, 20, 30]
weights = ["uniform", "distance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)

train_accuracies = [[] for _ in range(2)]
test_accuracies = [[] for _ in range(2)]

for k in K_VALUES:
    for i, weight in enumerate(weights):
        predictor = KNeighborsClassifier(n_neighbors = k, weights = weight)
        predictor.fit(X_train, y_train)

        train_accuracy = metrics.accuracy_score(y_train, predictor.predict(X_train))
        test_accuracy = metrics.accuracy_score(y_test, predictor.predict(X_test))

        train_accuracies[i].append(train_accuracy)
        test_accuracies[i].append(test_accuracy)

plt.figure(figsize=(12, 5))

train_color = "#4caf50"  
test_color = "#ff5722"   

for i, weight in enumerate(weights):
    plt.subplot(1, 2, i + 1)
    plt.plot(K_VALUES, train_accuracies[i], label = "Train Accuracy", marker = ".", color = train_color)
    plt.plot(K_VALUES, test_accuracies[i], label = "Test Accuracy", marker = "+",  color = test_color)
    plt.title(f"Train and Test Accuracy for kNN Classifier ({weight.capitalize()} Weights)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xticks(K_VALUES)

plt.tight_layout()
plt.show()
