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
from sklearn.linear_model import LinearRegression, Ridge

from tools.datasets import make_wave

X, y = make_wave(n_samples=180)
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


# create a model Ridge, train it on the same training
# set made of the Housing market and evaluate its
# training score and test score. Do you have any improvement ?
# Is it better compared to a model with no regularization?
model_ridge = Ridge()
model_ridge.fit(X_train, y_train)

model_ridge.score(X_test, y_test)
# not been finished yet
