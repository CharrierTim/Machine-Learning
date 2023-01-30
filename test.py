# Plot the maximum margin separating hyperplane within a two-class separable dataset using a Support Vector Machine classifier with linear kernel.

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

from sklearn.inspection import DecisionBoundaryDisplay

def SVC_C(C = [0.0001, 1, 100000], kernel = 'linear'):
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize data
    X_train = X_train / 255
    X_test = X_test / 255

    # PCA to reduce dimensionality
    # Reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Let's only take 2 classes
    X_train = X_train[y_train < 2]
    y_train = y_train[y_train < 2]

    X_test = X_test[y_test < 2]
    y_test = y_test[y_test < 2]

    # Plot the data on the same figure the 3 different result of the SVM
    figure = plt.figure(figsize=(10, 10))


    for c in C:
        clf = svm.SVC(kernel=kernel, C=c)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy with linear kernel and C = {}: ".format(c), accuracy_score(y_test, y_pred))

        # Plot each result on the same figure


        plt.subplot(2, 2, C.index(c) + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X_train,
            plot_method="contour",
            colors="k",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
            ax=ax,
        )
        # plot support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k",
        )

        plt.title("C = {}".format(c))
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")

    plt.show()

if __name__ == "__main__":
    SVC_C([0.00001, 1, 10])

    