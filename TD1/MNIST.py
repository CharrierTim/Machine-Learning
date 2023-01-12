# The dataset MNIST 

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.cluster import KMeans

# Load the dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Convert and normalize the data

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# Perform PCA to reduce the dimension of the data with 200 nb_components

pca = PCA(n_components=200).fit(X_train)
X_train = pca.transform(X_train)

# Perform K-means clustering with 10 clusters

kmn = KMeans(n_clusters=10, random_state=0).fit(X_train)
kmn.fit(X_train)

# Plot the results

y_pred = kmn.predict(X_train)

plt.figure(figsize=(10, 10))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred)
plt.show()