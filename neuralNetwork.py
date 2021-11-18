"""

The activation function  𝑓  of a neuron can be linear or non linear.
The most used activation functions are :

        sigmoid
        tanh
        ReLu (rectified linear)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from tools.plots import plot_2d_separator


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.linspace(-5, 5, 100)
sigmoid_x = sigmoid(x)
tanh_x = np.tanh(x)
ReLu_x = np.maximum(0, x)

figure = plt.figure()
plt.plot(x, sigmoid_x, label="sigmoid")
plt.plot(x, tanh_x, label="tanh")
plt.plot(x, ReLu_x, label="ReLu")
plt.legend()
plt.show()


X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

"""
As you can see, using a linear SVM does not seem to be the best choice. 
Let's use neural networks to be able to classify this dataset. 
First, split the data into a training and a test set.
"""
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=0)
network = MLPClassifier()
network.fit(X_train, y_train)
print("Train accuracy:", network.score(X_train, y_train))
print("Test  accuracy:", network.score(X_test, y_test))
plot_2d_separator(network, X, y)
plt.show()

"""
 the accuracy of the neural network is not very good (0.84) and with the decision boundary, 
 we can see that the network is not able to understand that the real decision boundary has a moon shape. 
 Moreover, we have a warning message indicating that the network has not converged after 200 iterations 
 (the default number of iterations). We need to tune our network to improve its performance.
"""

nn_adam = MLPClassifier(solver="adam")
nn_adam.fit(X_train, y_train)
print("Adam")
print("  Train accuracy:", nn_adam.score(X_train, y_train))
print("  Test  accuracy:", nn_adam.score(X_test, y_test))

nn_sgd = MLPClassifier(solver="sgd")
nn_sgd.fit(X_train, y_train)
print("SGD")
print("  Train accuracy:", nn_sgd.score(X_train, y_train))
print("  Test  accuracy:", nn_sgd.score(X_test, y_test))

nn_lbfgs = MLPClassifier(solver="lbfgs")
nn_lbfgs.fit(X_train, y_train)
print("LBFGS")
print("  Train accuracy:", nn_lbfgs.score(X_train, y_train))
print("  Test  accuracy:", nn_lbfgs.score(X_test, y_test))

"""
the 3 networks have the same test accuracy (0.84). 
The dataset is not large enough to make any conclusions 
on the performance increase/degradation when we swicth the training algorithm. 
But we can see that when using the LBFGS algorithm, we do not have any convergence warning message. 
As explained in the documentation, for small datasets LBFGS can converge faster.
"""

# by default, the neural network has only 1 hidden layer of
# 100 neurons. Use the right parameter to create a network of
# 2 hidden layers, each one having 10 neurons.

nn_2layers = MLPClassifier(hidden_layer_sizes=(10, 10),
                           max_iter=5000, random_state=0)
nn_2layers.fit(X_train, y_train)
print("2 hidden layers of 10 neurons")
print("  Train accuracy:", nn_2layers.score(X_train, y_train))
print("  Test  accuracy:", nn_2layers.score(X_test, y_test))

# Create other networks with different parameters to see if
# many small layers is better than 1 big layer.

nn_1 = MLPClassifier(hidden_layer_sizes=(1000),
                     max_iter=5000, random_state=0)
nn_1.fit(X_train, y_train)
print("1 hidden layer of 1000 neurons")
print("  Train accuracy:", nn_1.score(X_train, y_train))
print("  Test  accuracy:", nn_1.score(X_test, y_test))

nn_2 = MLPClassifier(hidden_layer_sizes=(5, 5, 5, 5),
                     max_iter=5000, random_state=0)
nn_2.fit(X_train, y_train)
print("4 hidden layers of 5 neurons")
print("  Train accuracy:", nn_2.score(X_train, y_train))
print("  Test  accuracy:", nn_2.score(X_test, y_test))

"""
we can see that with a single layer of 1000 neurons, 
the training and test accuracies are better than with 2 layers of 10 neurons.
 In general, the only way to increase the accuracy of the model is to increase the number of neurons, 
 because more neurons means that the model is more complex and can find non trivial boundaries. 
 But having more layers is more efficient than having more neurons. If the activation function is non-linear, 
 cascading this non-linearity can also create complex functions without requiring a lot of neurons. 
 We can see that with only 20 neurons divided into 4 layers (so 5 neurons per layer), 
 we obtain a better test accuracy than with 1000 neurons.

"""

# identity
nn_ident = MLPClassifier(hidden_layer_sizes=(20,20),
                         activation="identity",
                         max_iter=5000,
                         random_state=0)
nn_ident.fit(X_train, y_train)
print("Identity activation")
print("  Train accuracy:", nn_ident.score(X_train, y_train))
print("  Test  accuracy:", nn_ident.score(X_test, y_test))

# sigmoid
nn_sigm = MLPClassifier(hidden_layer_sizes=(20,20),
                        activation="logistic",
                        max_iter=5000,
                        random_state=0)
nn_sigm.fit(X_train, y_train)
print("Sigmoid activation")
print("  Train accuracy:", nn_sigm.score(X_train, y_train))
print("  Test  accuracy:", nn_sigm.score(X_test, y_test))

# tanh
nn_tanh = MLPClassifier(hidden_layer_sizes=(20,20),
                        activation="tanh",
                        max_iter=5000,
                        random_state=0)
nn_tanh.fit(X_train, y_train)
print("Tanh activation")
print("  Train accuracy:", nn_tanh.score(X_train, y_train))
print("  Test  accuracy:", nn_tanh.score(X_test, y_test))

# relu
nn_relu = MLPClassifier(hidden_layer_sizes=(20,20),
                        activation="relu",
                        max_iter=5000,
                        random_state=0)
nn_relu.fit(X_train, y_train)
print("ReLu activation")
print("  Train accuracy:", nn_relu.score(X_train, y_train))
print("  Test  accuracy:", nn_relu.score(X_test, y_test))

"""
in this case, ReLu activations are the best. 
This can be exlained with the graphic representations of the activation functions. 
During backpropagation, gradient is transfered layer from layer in reverse order, 
but the gradient is multiplied by the value of the derivative of the activation function everytime it is transfered. 
The derivative of a function at a certain point is equal to the slope of the function at this same point. 
With ReLu, the slope is very high compared to sigmoid or tanh. So the gradient have larger values, 
and the model can have a faster convergence, and so better results. In general, we need to test every activate function 
because the result depends on the dataset.

"""