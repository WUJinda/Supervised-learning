import graphviz
from sklearn.model_selection import train_test_split

from tools.plot_interactive_tree import plot_tree_progressive
import matplotlib.pyplot as plt
plot_tree_progressive()

# load the breast cancer dataset. Separate it into
# a training and a test file

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# create and train a decision tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
training_accuracy = tree.score(X_train, y_train)
test_accuracy = tree.score(X_test, y_test)
print("Training accuracy:", training_accuracy)
print("Test accuracy:", test_accuracy)

"""
we can see that the training accuracy is perfect (equal to 1) while the test accuracy is not.
This means that our decision tree model is overfitting (good on training set, bad on test set).
"""

# look at the documentation of DecisionTreeClassifier() and
# train a model named tree that only has a depth of 4
#?DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)

training_accuracy = tree.score(X_train, y_train)
test_accuracy = tree.score(X_test, y_test)
print("Training accuracy:", training_accuracy)
print("Test accuracy:", test_accuracy)
export_graphviz(tree, out_file="breast_tree.dot",
                class_names=["malignant", "benign"],
                feature_names=cancer.feature_names,
                impurity=False,
                filled=True)

with open("breast_tree.dot") as f:
    breast_tree = f.read()
graphviz.Source(breast_tree)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tree.feature_importances_, 'o')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.ylim(0,1)

