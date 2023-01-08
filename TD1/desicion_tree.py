# Decision tree classifier

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Importing the dataset
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape the data

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# define lists to collect scores
train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 20)]
# evaluate a decision tree for each depth
for i in values:
    # configure the model
    model = DecisionTreeClassifier(max_depth=i)
    # fit model on the training dataset
    model.fit(X_train_flat, Y_train)
    # evaluate on the train dataset
    train_yhat = model.predict(X_train_flat)
    train_acc = accuracy_score(Y_train, train_yhat)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    test_yhat = model.predict(X_test_flat)
    test_acc = accuracy_score(Y_test, test_yhat)
    test_scores.append(test_acc)
    # summarize progress
    print('Depth=%d, Train: %.3f, Test: %.3f' % (i, train_acc, test_acc))
   

# plot the depth vs accuracy
plt.plot(values, train_scores, '-o', label='Train', color='red')
plt.plot(values, test_scores, '-o', label='Test', color='blue')
plt.xlabel('Tree depth', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()
plt.show()







