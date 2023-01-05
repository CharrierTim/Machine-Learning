# The dataset MNIST 

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# Display the raw data

X = np.load('/home/tim/MNIST/MNIST_X_28x28.npy')
Y = np.load('/home/tim/MNIST/MNIST_y.npy')

print("X.shape = ", X.shape)
print("Y.shape = ", Y.shape)

# Display the picture with matplotlib in ../MNIST

nb_samples = random.randint(0, X.shape[0])

plt.imshow(X[nb_samples])
img_title = "This is a %i" %Y[nb_samples]
plt.title(img_title)
plt.show()
plt.clf()

# Split the data into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

print("X_train.shape = ", X_train.shape)
print("Y_train.shape = ", Y_train.shape)

# Function to report the distribution of the data in PERCENTAGE
def report_distribution(Y):
    nb_samples = Y.shape[0]
    for i in range(10):
        nb_i = np.sum(Y == i)
        print("There are %i samples of %i (%.2f %%) in the dataset" %(nb_i, i, nb_i/nb_samples*100))
        
report_distribution(Y_train)
report_distribution(Y_test)


