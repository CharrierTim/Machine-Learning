import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.datasets import fashion_mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils

# Load pre-shuffled MNIST data into train and test sets

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape the data to 28x28 pixels

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Convert the data to the right type

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
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))


# Compile the model

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model

model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=2, validation_data=(X_test, Y_test))


# Evaluate the model

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])

print('Test accuracy:', score[1])

# Evaluate how good the model is

print(model.metrics_names)
print(model.evaluate(X_test, Y_test, verbose=0))

# Check if the model is overfitting

print(model.metrics_names)
print(model.evaluate(X_train, Y_train, verbose=0))

# How



