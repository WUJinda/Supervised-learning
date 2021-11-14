from sklearn.model_selection import train_test_split

from tools.datasets import make_wave
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X, y = make_wave(400)
print("X has ", len(X), "points ")
# visualize the points with the following piece of code
plt.scatter(X, y)
plt.xticks([], [])
plt.show()
# Separate the dataset into a training part and a test part with random_state = 0.

# create train + test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# create regression models, train and evaluate
model_Reg = KNeighborsRegressor(n_neighbors=3)

model_Reg.fit(X_train, y_train)
accuracy_Reg = model_Reg.score(X_test, y_test)
print("The accuracy of model: ", accuracy_Reg)

neig_x, neig_y = [], []
for i in range(1, 27):
    neig_x.append(i)
    model_i = KNeighborsRegressor(n_neighbors=i)
    neig_y.append(model_i.fit(X_train, y_train).score(X_test, y_test))
plt.plot(neig_x, neig_y)
plt.show()
