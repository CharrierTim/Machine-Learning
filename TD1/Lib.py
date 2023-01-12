import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Importing the dataset
import psutil
from keras.datasets import mnist

# Neural network
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.datasets import fashion_mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten

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
# MLP class
############################################################################################################


class MLP:
    ''' This class creates a MLP model
    By @Timothée Charrier

    Parameters
    ----------
    epochs : int, see report for more details
    batch_size : int, see report for more details    
    '''

    def __init__(self, epochs=10, batch_size=128):
        self.X_train, self.Y_train, self.X_test, self.Y_test = Loading_MNIST().get_all_data()
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
        print('Total execution time: ', time_end - time_start, 's')
        print('Memory usage: ', psutil.Process(
            os.getpid()).memory_info().rss / 1024 ** 2, 'MB')

############################################################################################################
# CNN class using the Loading_MNIST class
############################################################################################################


class CNN:

    def __init__(self, epochs=10, batch_size=128):
        self.X_train, self.Y_train, self.X_test, self.Y_test = Loading_MNIST().get_all_data()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.history = None

    def create_model(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
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
        print('Total execution time: ', time_end - time_start, 's')
        print('Memory usage: ', psutil.Process(
            os.getpid()).memory_info().rss / 1024 ** 2, 'MB')


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


if __name__ == "__main__":
    mnist = Loading_MNIST()
    X_train, Y_train, X_test, Y_test = mnist.get_all_data()