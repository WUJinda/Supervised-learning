from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

model_breast = SVC(kernel="linear", C=10, random_state=0)
model_breast.fit(X_train, y_train)
print("Train accuracy:", model_breast.score(X_train, y_train))
print("Test  accuracy:", model_breast.score(X_test, y_test))

# test accuracy is better compared to a KNN model.