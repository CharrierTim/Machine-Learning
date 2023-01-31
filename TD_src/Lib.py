# Author: Timothée Charrier
# About this file: This file contains the functions used in the TD.

############################################################################################################
# Importing the libraries
############################################################################################################

# Ignore warnings
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import keras
import tensorflow as tf
from keras.datasets import mnist
import seaborn as sns
import sklearn.metrics as sk_metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Importing the libraries
# from sklearn.inspection import DecisionBoundaryDisplay

# Importing the dataset

# Neural network

############################################################################################################
# Loading_MNIST class
############################################################################################################


class Loading_MNIST:
    ''' This class loads the MNIST dataset and reshapes it to 4D
    By @Timothée Charrier
    '''

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Reshaphe the data to 4D
    def reshape_data(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    # Normalize the data
    def normalize_data(self):
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

    # Convert class vectors to binary class matrices
    def convert_to_binary(self):
        self.Y_train = keras.utils.to_categorical(self.Y_train, 10)
        self.Y_test = keras.utils.to_categorical(self.Y_test, 10)

    def __init__(self):
        self.reshape_data()
        self.normalize_data()
        self.convert_to_binary()

    def get_train_data_4D(self):
        return self.X_train, self.Y_train

    def get_test_data_4D(self):
        return self.X_test, self.Y_test

    def get_all_data_4D(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_all_data_raw(self):
        return mnist.load_data()

# MLP class using the Loading_MNIST class

############################################################################################################
# Plot confusion matrix
############################################################################################################


def show_confusion_matrix(test_labels, test_classes):

    ''' This function plots the confusion matrix of the model.
    
    Parameters
    ----------
    test_labels : array, the labels of the test set
    test_classes : array, the predicted classes of the test set

    Returns
    -------
    plot : plot of the confusion matrix
    '''

    # Compute confusion matrix and normalize
    plt.figure(figsize=(10, 10))
    confusion = sk_metrics.confusion_matrix(test_labels, test_classes)
    confusion_normalized = confusion / confusion.sum(axis=1)
    axis_labels = range(10)
    ax = sns.heatmap(
        confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
        cmap='Blues', annot=True, fmt='.4f', square=True)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

############################################################################################################
# MLP class
############################################################################################################


class MLP:

    ''' This class creates a MLP model
    By @Timothée Charrier

    Parameters
    ----------
    epochs : int, see report for more details **default = 10**
    batch_size : int, see report for more details **default = 128**
    '''

    def __init__(self, epochs=10, batch_size=128):
        self.X_train, self.Y_train, self.X_test, self.Y_test = Loading_MNIST().get_all_data_4D()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.history = None

    def create_model(self):
        self.model.add(Flatten(input_shape=(28, 28, 1)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

    def compile_model(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self):
        self.history = self.model.fit(self.X_train, self.Y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      verbose=1,
                                      validation_data=(self.X_test, self.Y_test))

    def evaluate_model(self):
        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model

    def run_model(self):
        time_start = time.time()
        self.create_model()
        self.compile_model()
        self.train_model()
        self.evaluate_model()
        time_end = time.time()
        print("Creating, compiling, training and evaluating the CNN model took",
              time_end - time_start, "seconds")

############################################################################################################
# CNN class using the Loading_MNIST class
############################################################################################################


class CNN:
    ''' This class creates a CNN model
    By @Timothée Charrier

    Parameters
    ----------
    epochs : int, see report for more details **default = 10**
    batch_size : int, see report for more details **default = 128**
    '''

    def __init__(self, epochs=10, batch_size=128):
        self.X_train, self.Y_train, self.X_test, self.Y_test = Loading_MNIST().get_all_data_4D()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.history = None

    def create_model(self):
        self.model.add(
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

    def compile_model(self):
        # Compiler should be adam
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self):
        self.history = self.model.fit(self.X_train, self.Y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      verbose=1,
                                      validation_data=(self.X_test, self.Y_test))

    def evaluate_model(self):
        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model

    def run_model(self):
        time_start = time.time()
        self.create_model()
        self.compile_model()
        self.train_model()
        self.evaluate_model()
        time_end = time.time()
        print("Creating, compiling, training and evaluating the CNN model took",
              time_end - time_start, "seconds")

    def plot_learning_curve(self):
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.plot(self.history.history['loss'], label='Training loss')
        plt.plot(self.history.history['val_loss'], label='Validation loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.xlabel('Epoch')
        plt.title('Training and Validation Loss')

        plt.subplot(2, 1, 2)
        plt.plot(self.history.history['accuracy'], label='Training accuracy')
        plt.plot(self.history.history['val_accuracy'],
                 label='Validation accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title('Training and Validation Accuracy')

        plt.show()


############################################################################################################
#  PCA functions
############################################################################################################

def PCA_reconstruction_example(X_train_flat, n_components):
    ''' This function displays the reconstruction of a random image from the MNIST dataset
    By @Timothée Charrier
    '''

    random_index = np.random.randint(0, X_train_flat.shape[0])

    List_explained_variance = []
    count = 0

    for n in n_components:
        pca = PCA(n_components=n)
        pca.fit(X_train_flat)

        #  Explained variance

        List_explained_variance.append(np.sum(pca.explained_variance_ratio_))

        # Reconstruct the data and reshape it with the PCA model and display a picture if n = 10, 50, 100, 200, 500, 784
        if (n == 10 or n == 50 or n == 100 or n == 200 or n == 500 or n == 784):
            X_train_reconstructed = pca.inverse_transform(
                pca.transform(X_train_flat))

            X_train_reconstructed = X_train_reconstructed.reshape(
                X_train_flat.shape[0], 28, 28)

            #  Display the reconstructed picture in the same figure

            plt.subplot(2, 3, count+1)
            plt.imshow(X_train_reconstructed[random_index], cmap='gray')
            plt.axis('off')
            count += 1

            if (n == 784):
                plt.title("Original")

            else:
                plt.title("Components = " + str(n))

    plt.suptitle(
        "Reconstruction of a random picture of the training set with different number of components")
    plt.show()
    return List_explained_variance

############################################################################################################
# SVC with different PCA components
############################################################################################################


def SVC_PCA(n_components):
    ''' This function creates a SVC model with different number of PCA components
    By @Timothée Charrier

    Parameters
    ----------
    n_components : array of int

    Returns
    -------
    List_accuracy_train : array
    List_accuracy_test : array
    '''

    #  Load the dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Reshaping the array for PCA
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    # Normalizing the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    #  Create a list of the accuracy of train set and test set
    List_accuracy_train = []
    List_accuracy_test = []

    #  Create a list of the time
    List_time = []

    # Perfom SVC on the dataset with diffrent PCA components
    for n in n_components:
        #  Create a PCA object with n components
        pca = PCA(n_components=n)

        #  Fit the PCA object on the train set
        pca.fit(X_train)

        #  Transform the train set and test set
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        #  Create a SVC object
        svc = SVC()

        #  Fit the SVC object on the train set
        svc.fit(X_train_pca, Y_train)

        #  Get the accuracy of the train set and test set
        accuracy_train = svc.score(X_train_pca, Y_train)
        accuracy_test = svc.score(X_test_pca, Y_test)

        #  Append the accuracy to the list
        List_accuracy_train.append(accuracy_train)
        List_accuracy_test.append(accuracy_test)

    #  Plot the accuracy of train set and test set
    plt.plot(n_components, List_accuracy_train, label='Train set')
    plt.plot(n_components, List_accuracy_test, label='Test set')
    plt.xlabel('Number of components')
    plt.ylabel('Accuracy')
    plt.title(
        'Accuracy of train set and test set with different number of components')
    plt.legend()
    plt.show()

    return List_accuracy_train, List_accuracy_test

############################################################################################################
#  Function to compare SVM with different kernels
############################################################################################################


def SVC_kernels(K=['linear', 'poly', 'rbf', 'sigmoid']):
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize data
    X_train = X_train / 255
    X_test = X_test / 255

    # PCA to reduce dimensionality
    from sklearn.decomposition import PCA
    pca = PCA(n_components=200)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Kernel functions
    Acc_list = []

    # Train SVM with different kernels
    for k in K:
        clf = svm.SVC(kernel=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        Acc_list.append(accuracy_score(y_test, y_pred))

    return Acc_list

############################################################################################################
#  Function to compare SVM with different C values
############################################################################################################


def SVC_C(C=[0.0001, 1, 100000], kernel='linear'):

    print("/!\")")
    print("THIS FUNCTION WON'Y WORK ON GOOGLE COLAB")
    print("It uses DecisionBoundary function, which is not a on a stable realease of scikit-learn yet")
    print("/!\")")

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
        print("Accuracy with linear kernel and C = {}: ".format(
            c), accuracy_score(y_test, y_pred))

        # Plot each result on the same figure

        plt.subplot(2, 2, C.index(c) + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.scatter(X_train[:, 0], X_train[:, 1],
                    c=y_train, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        # DecisionBoundaryDisplay.from_estimator(clf,X_train,plot_method="contour",colors="k",levels=[-1, 0, 1],alpha=0.5,linestyles=["--", "-", "--"],ax=ax,)
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


############################################################################################################
#  Main
############################################################################################################

__func__ = "CNN"

if __name__ == "__main__":
    if __func__ == "MLP":
        model = MLP()
        model.run_model()
    elif __func__ == "CNN":
        model = CNN()
        model.run_model()
    else:
        mnist = Loading_MNIST()
