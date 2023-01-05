# perform a supervised classification task using Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# The dataset MNIST

X = np.load('/home/tim/MNIST/MNIST_X_28x28.npy')
Y = np.load('/home/tim/MNIST/MNIST_y.npy')

# Split the data into training and testing and flatten the data to 2D

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Data classification using Logistic Regression with different solvers and plot the results on the same graph

def logistic_regression_function():
    
    solvers = ['sag', 'lbfgs']
    
    for solver in solvers:
        
        # Data classification using Logistic Regression
        
        logistic_regression = LogisticRegression(solver=solver, multi_class='multinomial', max_iter=1000).fit(X_train_flat, Y_train)
        
        # Use PCA to reduce the dimensionality of the data to 2D
        
        pca = PCA(n_components=2)
        pca.fit(X_train_flat)
        X_train_pca = pca.transform(X_train_flat)
        
        # Plot the data in 2D
        
        plt.subplot(1, 5, solvers.index(solver) + 1)
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=logistic_regression.predict(X_train_flat))
        plt.title("solver = " + solver)
        
        print("-----------------")
        print("solver = ", solver)
        print("Score = ", logistic_regression.score(X_train_flat, Y_train))
        print("-----------------")
        
    plt.show()
    
logistic_regression_function()