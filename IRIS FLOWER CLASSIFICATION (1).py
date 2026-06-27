
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Iris Flower Classification")
print("--------------------------")
print("Accuracy:", accuracy * 100, "%")

# Display the first 5 predictions
for i in range(5):
    print("Predicted:", iris.target_names[predictions[i]],
          "| Actual:", iris.target_names[y_test[i]])

# Plot a simple graph
plt.figure(figsize=(6,4))
plt.bar(["Accuracy"], [accuracy * 100])
plt.title("Iris Flower Classification Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.show()
