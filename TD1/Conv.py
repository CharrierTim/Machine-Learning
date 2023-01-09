# Convolutional Neural Network

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Import SCV and PCA

from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Importing the dataset
from keras.datasets import mnist

import matplotlib.pyplot as plt


# Importing the dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping the array for PCA

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Normalizing the data

X_train = X_train / 255.0
X_test = X_test / 255.0

# Perfom SVC on the dataset with PCA = 10

pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)

# Fitting SVM to the Training set

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results

X_test = pca.transform(X_test)
y_pred = classifier.predict(X_test)

# Accuracy

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))