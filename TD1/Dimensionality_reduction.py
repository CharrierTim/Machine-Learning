# Dimensionality reduction using sklearn PCA decomposition
# By Timothée Charrier and Thomas Ravelet

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# The dataset MNIST

X = np.load('/home/tim/MNIST/MNIST_X_28x28.npy')
Y = np.load('/home/tim/MNIST/MNIST_y.npy')

# Split the data into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

# Dimensionality reduction using sklearn PCA decomposition
# Flatten the data to 1D

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

#print("X_train_flat.shape = ", X_train_flat.shape)
#print("X_test_flat.shape = ", X_test_flat.shape)

# Apply PCA with different values of n_components and display a random MNIST pictures with different values of n_components
def plot_pca():
    n_components = [784, 10, 50, 100, 200, 500]
    random_index = np.random.randint(0, X_train.shape[0])

    print("-----------------")

    for n in n_components:
        pca = PCA(n_components=n)
        pca.fit(X_train_flat)
        print("n_components = ", n)
        print("Sum = ", np.sum(pca.explained_variance_ratio_))
        print("-----------------")

        # Reconstruct the data and reshape it with the PCA model and display a picture

        X_train_reconstructed = pca.inverse_transform(
            pca.transform(X_train_flat))

        X_train_reconstructed = X_train_reconstructed.reshape(
            X_train.shape[0], 28, 28)

        # For each n_components, display the first picture of the training set in the same figure

        plt.subplot(2, 3, n_components.index(n) + 1)
        plt.imshow(X_train_reconstructed[random_index], cmap='gray')

        if(n == 784):
            plt.title("Original")

        else:
            plt.title("Components = " + str(n))

        plt.axis('off')
    plt.show()


# Main

plot_pca()