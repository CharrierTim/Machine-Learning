
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = np.load('/home/tim/MNIST/MNIST_X_28x28.npy')
Y = np.load('/home/tim/MNIST/MNIST_y.npy')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

def SVM(kernel):
    clf = SVC(kernel=kernel)
    clf.fit(X_train_flat, Y_train)
    Y_pred = clf.predict(X_test_flat)
    print("Accuracy with %s kernel = %.2f %%" %(kernel, accuracy_score(Y_test, Y_pred)*100))
    
SVM("linear")
    
