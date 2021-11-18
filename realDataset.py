"""
Sklearn 附带了一些真实案例数据集。其中之一是威斯康星州乳腺癌数据集。
它包含乳腺癌肿瘤的信息（测量值）。每个肿瘤要么是“良性”要么是“恶性”
（因此它是一个二元分类问题）。我们将使用 KNN 来预测肿瘤是“良性”还是“恶性”。
"""
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from tools.datasets import load_extended_boston

cancer = load_breast_cancer()
# print(cancer.DESCR) # uncomment for more information
print(cancer.keys())
print(cancer.data.shape)
print(cancer.data[0])
print(cancer.target[0])
# Separate the points into a training and a test datasets with random_state = 0.
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# Create a KNN classifier with six neighbors and train it with the appropriate data.
model_breast = KNeighborsClassifier(n_neighbors=6)
model_breast.fit(X_train, y_train)
# use the .predict() method of your classifier and feed it with one or more data points
y_pred = model_breast.predict(X_test)
# Now compute the accuracy of your model on the entire test dataset.
accuracy_breast = model_breast.score(X_test, y_test)
print("The accuracy of breast model: ", accuracy_breast)

"""
The data come from the housing market in Boston. 
We have 506 data points, and each one has 104 features.
"""

X, y = load_extended_boston()
boston = load_boston()
print(boston.keys())
print(X.shape)
print(y.shape)
print(y[:3]) # some house prices
# separate the data into a training set and a testing set
# with random_state = 0. Then train a linear model and
# predict the price of the first house in the test set.
# Compare it with the actual price of the house.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
model_house = LinearRegression()
model_house.fit(X_train, y_train)
prediction = model_house.predict([X_test[0]])[0]
print("Predicted price =", prediction)
print("Real price =", y_test[0])
print("Difference between prediction and real price:",
      round((prediction-y_test[0]) / y_test[0] * 100, 2), "%")

print(model_house.score(X_train, y_train))
print(model_house.score(X_test, y_test))



