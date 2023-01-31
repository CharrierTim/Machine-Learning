# Let's make a CNN overfit
# By: Timothée Charrier

# Importing the libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import random
import time
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
import sklearn.metrics as sk_metrics
import seaborn as sns

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
# Overfitting
############################################################################################################

def Overfit_few_samples(nb_samples = 10):

    ''' This function creates a CNN model with only one hidden layer and trains it on a small number of samples.
    The goal is to overfit the model and see how it performs on the test set.

    Parameters
    ----------
    nb_samples : int, the number of samples to use for training

    Returns
    -------
    plot : plot of the accuracy and loss of the model on the training and validation sets and confusion matrix
    print : the accuracy of the model on the test set
    '''

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Reshape the data to 28x28 pixels

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize the data to the range of [0, 1]

    X_train /= 255
    X_test /= 255

    # Convert the labels to one-hot encoding

    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)

    # Take only 10 images for training

    X_train = X_train[:nb_samples]
    Y_train = Y_train[:nb_samples]

    batch_size = 128
    epochs = 100

    # Create the model with only one hidden layer

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose =0)

    # Evaluate the model

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot images from the predictions

    predictions = model.predict(X_test, verbose=0)

    # Plot learning curves

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Accuracy')

    # Confusion matrix
    show_confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(predictions, axis=1))


def Overfit_huge_batch_size(batch_size = 8192):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Reshape the data to 28x28 pixels

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize the data to the range of [0, 1]

    X_train /= 255
    X_test /= 255

    # Convert the labels to one-hot encoding

    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)

    batch_size = 8192
    epochs = 10

    # Create the model with only one hidden layer

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose = 0)

    # Evaluate the model

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot images from the predictions

    predictions = model.predict(X_test)
    # Plot learning curves

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Accuracy')

    # Confusion matrix
    show_confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(predictions, axis=1))

if __name__ == "__main__":
    print("Overfit few samples")