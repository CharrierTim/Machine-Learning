# Data clustering using k-means and EM with Gaussian Mixture Models
# By Timothée Charrier and Thomas Ravelet

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# The dataset MNIST

X = np.load('/home/tim/MNIST/MNIST_X_28x28.npy')
Y = np.load('/home/tim/MNIST/MNIST_y.npy')

# Split the data into training and testing and flatten the data to 2D

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


def Kmean_function():
    n_clusters = [2, 5, 10]

    for n in n_clusters:

        # Data clustering using k-means

        kmeans = KMeans(n_clusters=n, random_state=0).fit(X_train_flat)

        # Use PCA to reduce the dimensionality of the data to 2D

        pca = PCA(n_components=2)
        pca.fit(X_train_flat)
        X_train_pca = pca.transform(X_train_flat)

        # Plot the data in 2D

        plt.subplot(1, 3, n_clusters.index(n) + 1)
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=kmeans.labels_)
        plt.title("n_clusters = " + str(n))

        print("-----------------")
        print("n_clusters = ", n)
        print("Score = ", kmeans.score(X_train_flat))
        print("-----------------")

    plt.show()


def Kmean_3D_function():

    # Data clustering using k-means

    kmeans = KMeans(n_clusters=10, random_state=0).fit(X_train_flat)

    # Use PCA to reduce the dimensionality of the data to 3D

    pca = PCA(n_components=3)
    pca.fit(X_train_flat)
    X_train_pca = pca.transform(X_train_flat)

    # Plot the data in 3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
               X_train_pca[:, 2], c=kmeans.labels_)
    plt.show()

    print("Score = ", - kmeans.score(X_train_flat))
    print("Evaluation = ", kmeans.inertia_)


def EM_function():

    # Data clustering using EM with Gaussian Mixture Models

    EM = GaussianMixture(n_components=10, random_state=0).fit(X_train_flat)

    # Use PCA to reduce the dimensionality of the data to 2D

    pca = PCA(n_components=2)
    pca.fit(X_train_flat)
    X_train_pca = pca.transform(X_train_flat)

    # Plot the data in 2D

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                c=EM.predict(X_train_flat))
    plt.show()

    print("Score = ", EM.score(X_train_flat))
    print("Evaluation = ", EM.bic(X_train_flat))


# Main

EM_function()