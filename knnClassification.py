from tools.datasets import make_forge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from tools.plots import plot_2d_separator

x, y = make_forge()

print(x[:5])
print(y[:5])
# If needed, recreate X and y so you have 600 data points.
x, y = make_forge(600)
print(x[:5])
print(y[:5])
# print the points X with matplotlib


plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
"""
Learning a model
As we saw in the course, the first step is to separate our dataset into a training and a test part. Use the function train_test_split() to create four variables :

points for training
labels for training
points for test
labels for test
Use the parameter random_state = 0 so the experiments can be replicated.

"""


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
print("Total number of points:", len(x))
print("Nmuber of points for training:", len(X_train))
print("Nmuber of points for test:", len(X_test))

# Then, we can create a KNN model and specify the parameter k. Create a model with k = 3.
model = KNeighborsClassifier(n_neighbors=3)
# Train the model on your training data (with the .fit() method) and evaluate its performance (with the .score() method) on the test data.
model.fit(X_train,y_train)

accuracy = model.score(X_test, y_test)
print("accuracy = ",  accuracy)
print("# Misclassified points = ", int(round((1-accuracy)*len(X_test))))

plot_2d_separator(model, x, y, fill=True, eps=0.5, alpha=0.4)
plt.title("Decision boundary when k = 3")

# models with k = 1, 9, 15
model1 = KNeighborsClassifier(n_neighbors=1)
model1.fit(X_train,y_train)

accuracy1 = model1.score(X_test, y_test)
accuracy_t = model1.score(X_train, y_train)
print("accuracy1 = ",  accuracy1)
print("accuracy train = ",  accuracy_t)
# decision boundary for each models
print("# Misclassified points = ", int(round((1-accuracy1)*len(X_test))))
plot_2d_separator(model1, x, y, fill=True, eps=0.5, alpha=0.4)
plt.title("Decision boundary when k = 1")

model9 = KNeighborsClassifier(n_neighbors=9)
model9.fit(X_train,y_train)

accuracy9 = model9.score(X_test, y_test)
print("accuracy9 = ",  accuracy9)
print("# Misclassified points = ", int(round((1-accuracy9)*len(X_test))))
plot_2d_separator(model9, x, y, fill=True, eps=0.5, alpha=0.4)
plt.title("Decision boundary when k = 9")

model15 = KNeighborsClassifier(n_neighbors=15)
model15.fit(X_train,y_train)

accuracy15 = model15.score(X_test, y_test)
print("accuracy15 = ",  accuracy15)
print("# Misclassified points = ", int(round((1-accuracy15)*len(X_test))))
plot_2d_separator(model15, x, y, fill=True, eps=0.5, alpha=0.4)
plt.title("Decision boundary when k = 15")

