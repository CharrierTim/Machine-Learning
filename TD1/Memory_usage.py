
# Importing the libraries
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

batch_size = 128
epochs = 10

def Memory_usage_CNN():
    time1 = time.time()
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

    # Create the model with only one hidden layer

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=0)

    # Evaluate the model

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    time2 = time.time()
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/1000000, time2 - time1


def Memory_usage_MLP():
    time1 = time.time()
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

    # Create the model with only one hidden layer
    # Create the model

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))


    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=0)


    # Evaluate the model

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    time2 = time.time()
    process = psutil.Process(os.getpid())

    return process.memory_info().rss/1000000, time2 - time1

def Average_Memory_usage_MLP():
    memory_usage_MLP_list = []
    execution_time_MLP_list = []

    for i in range(10):
        memory_usage_MLP, execution_time_MLP = Memory_usage_MLP()
        memory_usage_MLP_list.append(memory_usage_MLP)
        execution_time_MLP_list.append(execution_time_MLP)

    print("Average Memory Usage on 10 samples for MLP: ", np.mean(memory_usage_MLP_list))
    print("Average Execution Time on 10 samples for MLP: ", np.mean(execution_time_MLP_list))

if __name__ == '__main__':
    Average_Memory_usage_MLP()






