"""
Ordinary Least Squares
    普通最小二乘法是最经典的线性回归方法。
    该模型找到了 w 和 b 参数，它们最小化了预测
    与训练数据集中 𝑚 点的真实值之间的均方误差 (MSE)。
    <script type="math/tex; mode=display" id="MathJax-Element-2">
    \begin{equation*}\hat{y} = \sum_{k=1}^n w_k \times x_k + b\end{equation*}
    </script>


"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression

from tools.datasets import make_wave, make_forge
import matplotlib.pyplot as plt
import pandas as np

from tools.plots import plot_2d_separator

X, y = make_wave(n_samples=180)
print("X has", len(X), "points.")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# create model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# The learned  𝑤  are in the coef_ attribute while the learned  𝑏  are in the intercept_ attribute.
# Since our data only has one feature, we only have one  𝑤 .
print("Learned w:", model.coef_)
print("Learned b:", model.intercept_)

print("Model prediction =", model.predict([X_test[0]]))
print("Hand computed prediction =", model.coef_[0] * X_test[0] + model.intercept_)
print("Correct output =", y_test[0])

print("Score for linear reg", model.score(X_test, y_test))

"""
Regularization
    有时，线性模型可能会过度拟合。这意味着它在训练集上会很好，
    但在测试集上不会。控制过度拟合的一种方法是向我们的模型添加正则化。
    我们可以为模型最小化的目标添加一个约束。
    我们将看到 L2 归一化最小化模型权重 𝑤 的范数 2。
    这种新型模型的名称称为岭回归( Ridge regression )，它最小化：

    
    MSE + Regularization = {1 \over {m}} \sum_{k=1}^m (\hat{y}-y)^2 + \lambda \left\lVert w \ right\ rVert ^2
    𝜆 是调整正则化效果的参数。

"""

model_ridge = Ridge()
model_ridge.fit(X_train, y_train)
print("Training score R^2 =", model_ridge.score(X_train, y_train))
print("Test score R^2 =", model_ridge.score(X_test, y_test))

model_ridge_01 = Ridge(alpha=0.1)
model_ridge_01.fit(X_train, y_train)
print("Training coefficient R^2 (alpha=0.1) =", model_ridge_01.score(X_train, y_train))
print("Test coefficient R^2     (alpha=0.1) =", model_ridge_01.score(X_test, y_test))

model_ridge_02 = Ridge(alpha=0.2)
model_ridge_02.fit(X_train, y_train)
print("Training coefficient R^2 (alpha=0.2) =", model_ridge_02.score(X_train, y_train))
print("Test coefficient R^2     (alpha=0.2) =", model_ridge_02.score(X_test, y_test))

model_ridge_05 = Ridge(alpha=0.5)
model_ridge_05.fit(X_train, y_train)
print("Training coefficient R^2 (alpha=0.5) =", model_ridge_05.score(X_train, y_train))
print("Test coefficient R^2     (alpha=0.5) =", model_ridge_05.score(X_test, y_test))

model_ridge_2 = Ridge(alpha=2)
model_ridge_2.fit(X_train, y_train)
print("Training coefficient R^2 (alpha=2)   =", model_ridge_2.score(X_train, y_train))
print("Test coefficient R^2     (alpha=2)   =", model_ridge_2.score(X_test, y_test))

"""
当 alpha（公式中的 lambda）增加时，正则化更有效，
因此训练和测试系数 R² 之间的差距减小。当我们增加 alpha 时，我们正在防止过度拟合。 
但是当 alpha 太大时，模型会尝试将 W 的范数最小化而不是 MSE。性能开始变差。
主要思想是找到使我们模型性能最大化的最佳 alpha。
"""

"""
Linear Models for classification

"""
# use the make_forge() function to generate a set of 500
# points X and their labels y for classification.

# We can pass a parameter to make_forge() to indicate the
# number of points we want.
X, y = make_forge(500)
print("X has", len(X), "points.")

# this snippet prints the points on a 2d space


plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("Training set has", len(X_train), "points.")
print("Test     set has", len(X_test), "points.")

# then we can create a model
model = LogisticRegression(solver="lbfgs")
model.fit(X_train, y_train)
print("Model accuracy:", model.score(X_test, y_test))
plot_2d_separator(model, X, y, fill=True, eps=0.5, alpha=0.4)
#plt.savefig("decision-boundary.png")
plt.show()

# Compute number of misclassified points:

# First test if the predicted class is equal to the true
# label. We get an array of booleans, each one indicates if
# test_point[i] is misclassified.
is_misclassified = model.predict(X_test) != y_test
y_pred = model.predict(X_test)
proba_predict = model.predict_proba(X_test)

print(proba_predict[0])
print(y_pred[0])

# Then, we use the fact that True is evaluated as 1 and False
# as 0. So if we sum all booleans, we get the number of times
# there is a True value in the array, which is also the
# number of misclassified points.
n_misclassified = sum(is_misclassified)
print("Number of misclassified points:", n_misclassified)

# Create some LogisticRegression() models with regularization
# Visualize their respective decision boundary

# instead of saving each images, I plot the decision
# boundaries in a 2x2 subplots figure (easier to see the
# differences between boundaries)

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

# model 1
c = 0.0002
model_1 = LogisticRegression(solver="lbfgs", C=c)
model_1.fit(X_train, y_train)
acc = model_1.score(X_test, y_test)
print("Model 1")
print("  C =", c)
print("  Accuracy = ", acc)
print("  Misclassified = ",
      sum(model_1.predict(X_test) != y_test))
plot_2d_separator(model_1, X, y, ax=ax[0][0],
                  fill=True, eps=0.5, alpha=0.4)
ax[0][0].set_title("C = {} / Acc = {}".format(c, acc))

# model 2
c = 0.0005
model_2 = LogisticRegression(solver="lbfgs", C=c)
model_2.fit(X_train, y_train)
acc = model_2.score(X_test, y_test)
print("Model 2")
print("  C =", c)
print("  Accuracy = ", acc)
print("  Misclassified = ",
      sum(model_2.predict(X_test) != y_test))
plot_2d_separator(model_2, X, y, ax=ax[0][1],
                  fill=True, eps=0.5, alpha=0.4)
ax[0][1].set_title("C = {} / Acc = {}".format(c, acc))

# model 3
c = 0.001
model_3 = LogisticRegression(solver="lbfgs", C=c)
model_3.fit(X_train, y_train)
acc = model_3.score(X_test, y_test)
print("Model 3")
print("  C =", c)
print("  Accuracy = ", acc)
print("  Misclassified = ",
      sum(model_3.predict(X_test) != y_test))
plot_2d_separator(model_3, X, y, ax=ax[1][0],
                  fill=True, eps=0.5, alpha=0.4)
ax[1][0].set_title("C = {} / Acc = {}".format(c, acc))

# model 4
c = 0.01
model_4 = LogisticRegression(solver="lbfgs", C=c)
model_4.fit(X_train, y_train)
acc = model_4.score(X_test, y_test)
print("Model 4")
print("  C =", c)
print("  Accuracy = ", acc)
print("  Misclassified = ",
      sum(model_4.predict(X_test) != y_test))
plot_2d_separator(model_4, X, y, ax=ax[1][1],
                  fill=True, eps=0.5, alpha=0.4)
ax[1][1].set_title("C = {} / Acc = {}".format(c, acc))
plt.show(fig)

print("Model 1")
print("  w =", model_1.coef_)
print("  b =", model_1.intercept_)
print("Model 2")
print("  w =", model_2.coef_)
print("  b =", model_2.intercept_)
print("Model 3")
print("  w =", model_3.coef_)
print("  b =", model_3.intercept_)
print("Model 4")
print("  w =", model_4.coef_)
print("  b =", model_4.intercept_)

"""
Compare the images of decision boundaries you have saved. Can you see the influence of the regularization parameter ? 
Can this parameter help to prevent overfitting or underfitting ?

Answer : the parameter C is the inverse of the regularization strength. 
This means that when C=0.0002, the regularization is stronger than when C=0.01. 
With the decision boundaries, we can see that when the regularization is weak (C=0.01),
 the decision boundary is close to our data (the direction of the boundary is almost 
 the same as the data, like a \ line) but when it is strong (C=0.0002), 
 the boundary does not align with the data (the line is more horizontal and is shifted upwards). 
 Like we said before, we need to find a C that balances the effect of the regularization. 
 In our case, the best accuracy is obtained with C=0.0005.
 
 
 Does the regularization parameter has an influence on the values of theses coefficients ?
 
 Answer : yes, C has an effect on the values of  𝑤  and  𝑏 . 
 We can see that, the stronger the regularization, 
 the lower the values of  𝑤  and  𝑏 . This is what we were expecting 
 because the regularization tries to minimize the norm  ‖𝑊‖ . 
 So if the model needs to optimize a lot the norm of  𝑊 , 
 the only option is to decrease the values of  𝑤  and  𝑏 .
"""


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
c = 0.0005
model = LogisticRegression(solver="lbfgs", C=c)
model.fit(X_train, y_train)
print("Model accuracy:", model.score(X_test, y_test))
