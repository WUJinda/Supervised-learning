# Create a dataset of 300 points with make_forge()
# and split it into a 270 points training set and
# 30 points test set
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tools.datasets import make_forge
from tools.plots import plot_2d_separator

X, y = make_forge(300)
print("X has ", len(X), " points.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=30, random_state=0)
print("X_train has", len(X_train), "points.")
print("X_test has", len(X_test), "points.")

# Print the training points on a 2d figure.

colors_train = np.where(y_train == 1, "salmon", "lightblue")
plt.scatter(X_train[:, 0], X_train[:, 1], c=colors_train, s=10)
colors_test = np.where(y_test == 1, "orange", "blue")
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors_test, s=10)

plt.show()

# Create a SVM model for classification with SVC class.
# Use a linear kernel. Train it and evaluate its accuracy
model = SVC(kernel="linear")
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Accuracy of linear SVC =", accuracy)

plot_2d_separator(model, X_test, y_test)
plt.show()

n_misclassified = sum(model.predict(X_test) != y_test)
print("{}/{} misclassified points".format(
    n_misclassified, len(X_test)))

model_1 = SVC(kernel="linear", C=100)
model_1.fit(X_train, y_train)
print("Model 1 (C=100)")
print("  Train accuracy =", model_1.score(X_train, y_train))
print("  Test  accuracy =", model_1.score(X_test, y_test))
plot_2d_separator(model_1, X_test, y_test)
plt.scatter(model_1.support_vectors_[:, 0],
            model_1.support_vectors_[:, 1], c="lime", s=2)
plt.show()

model_2 = SVC(kernel="linear", C=10)
model_2.fit(X_train, y_train)
print("Model 2 (C=10)")
print("  Train accuracy =", model_2.score(X_train, y_train))
print("  Test  accuracy =", model_2.score(X_test, y_test))
plot_2d_separator(model_2, X_test, y_test)
plt.scatter(model_2.support_vectors_[:, 0],
            model_2.support_vectors_[:, 1], c="lime", s=2)
plt.show()

model_3 = SVC(kernel="linear", C=0.1)
model_3.fit(X_train, y_train)
print("Model 3 (C=0.1)")
print("  Train accuracy =", model_3.score(X_train, y_train))
print("  Test  accuracy =", model_3.score(X_test, y_test))
plot_2d_separator(model_3, X_test, y_test)
plt.scatter(model_3.support_vectors_[:, 0],
            model_3.support_vectors_[:, 1], c="lime", s=2)
plt.show()

model_4 = SVC(kernel="linear", C=0.01)
model_4.fit(X_train, y_train)
print("Model 4 (C=0.01)")
print("  Train accuracy =", model_4.score(X_train, y_train))
print("  Test  accuracy =", model_4.score(X_test, y_test))
plot_2d_separator(model_4, X_test, y_test)
plt.scatter(model_4.support_vectors_[:, 0],
            model_4.support_vectors_[:, 1], c="lime", s=2)
plt.show()

model_5 = SVC(kernel="linear", C=0.001)
model_5.fit(X_train, y_train)
print("Model 5 (C=0.001)")
print("  Train accuracy =", model_5.score(X_train, y_train))
print("  Test  accuracy =", model_5.score(X_test, y_test))
plot_2d_separator(model_5, X_test, y_test)
plt.scatter(model_5.support_vectors_[:, 0],
            model_5.support_vectors_[:, 1], c="lime", s=2)
plt.show()

"""
 the parameter C represents the weight of each misclassified point. 
 When C is high, the model tries to avoid as much as possible a misclassified point, 
 even if it implies to have a smaller margin. We can see that when C=100 or when C=10, 
 the number of support vectors (in green) is small, and the distance between 
 the black line and the furthest support vector is low (this distance is the margin).

When C is low, the model tries to have a large margin, instead of trying to reduce 
the number of misclassified points (when C is low, the weight of an error is also low). 
We can see that with C=0.001, the number of support vectors is high (a lot of green points) 
and so is the margin. In this case, the test accuracy is perfect.

The complexity of a SVM can be represented by the number of support vectors needed to 
draw the separating line. We can see that when C is low, there are a lot of support vectors, 
so our model is complex. And because we have a perfect test accuracy, we can suppose 
that our model is overfitting (too close to our dataset, not very able to generalize to unseen new data points). 
By increasing the value of C, we reduce the number of required support vectors, so is the complexity of our model. 
We are less likely to be overfitting.

"""

