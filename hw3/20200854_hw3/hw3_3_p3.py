"""
Filename: hw3_3_p3.py
Author: Quang Minh Nguyen

Submission for homework 3, task 3b
References: MMDS book, Section 12.3.4
"""

import sys
import numpy as np 
from time import time


# hyperparameters
N_ITERS = 100 # number of iterations


def preprocess(features, labels):
    """
    Preprocess two files
        - features: each line contains values 
        of features of a single data point 
        separated by commas
        - labels: each line contains the label of a 
        single data point, either 1 or -1
    
    and return an array where each row is a data point
    and another array where each entry is the label of
    the corresponding data point.

    Parameters:
        file: str, path to the data file
    
    Returns:
        X: ndarray of shape (N, D), each row is a data 
            point augmented by a 1
        Y: ndarray of shape (D,), each entry is the label 
            of the corresponding data point
    """
    X = np.loadtxt(features, delimiter=",")
    # deletes zero columns from X
    X = X[:, np.any(X, axis=0)]
    # normalises the data
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    Y = np.loadtxt(labels)
    return X, Y


def train(X, Y, c, eta):
    """
    Fit an SVM on features X and labels Y using 
    batch gradient descent and return the weights.

    Parameters:
        X: ndarray, each row is a data point
            augmented by a 1
        Y: ndarray, each entry is the label of the
            corresponding data point
    
    Returns:
        W: ndarray, the weights of the SVM
    """
    W = 0.01*np.random.randn(X.shape[1])
    for _ in range(N_ITERS):
        # computes gradient
        gradient = np.zeros(X.shape[1])
        for i in range(len(X)):
            if Y[i]*(np.dot(W, X[i])) < 1:
                gradient -= Y[i]*X[i]
        gradient = W + c*gradient
        # updates weights
        W -= eta*gradient
    return W


def predict(W, X):
    """
    Predict labels for data points in X using
    the weights W.

    Parameters:
        W: ndarray, the weights of the SVM
        X: ndarray, each row is a data point
            augmented by a 1

    Returns:
        Y: ndarray, each entry is the predicted
            label of the corresponding data point
    """
    Y = np.sign(np.dot(X, W))
    return Y


def accuracy(predictions, Y):
    """
    Compute the accuracy of predictions.

    Parameters:
        predictions: ndarray, the predicted labels
        Y: ndarray, the true labels
    
    Returns:
        acc: float, the accuracy
    """
    acc = np.sum(predictions == Y)/Y.shape[0]
    return acc


def get_cross_validation_accuracy(X, Y, c, eta):
    """
    Compute the cross validation accuracy of the SVM
    using 10-fold cross validation

    Parameters:
        X: ndarray, each row is a data point
            augmented by a 1
        Y: ndarray, each entry is the label of the
            corresponding data point
        c: float, the regularisation parameter
        eta: float, the learning rate
    
    Returns:
        acc: float, the cross validation accuracy
    """

    # shuffles data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # splits data into 10 folds
    X_folds = np.split(X, 10)
    Y_folds = np.split(Y, 10)

    # computes cross validation accuracy
    acc = 0
    for i in range(10):
        # gets training folds
        X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
        Y_train = np.concatenate(Y_folds[:i] + Y_folds[i+1:])
        # gets validation fold
        X_val = X_folds[i]
        Y_val = Y_folds[i]
        # trains SVM and gets weights
        W = train(X_train, Y_train, c, eta)
        # computes accuracy
        predictions = predict(W, X_val)
        acc += accuracy(predictions, Y_val)
    acc /= 10
    return acc


def grid_search(X, Y):
    """
    Perform grid search to find the best hyperparameters.

    Returns:
        best_c: float, the best regularisation parameter
        best_eta: float, the best learning rate
        best_acc: float, the best cross validation accuracy
    """
    
    # hyperparameters
    Cs      = np.arange(0.1, 1, 0.2)  # 5 values 
    ETAs    = np.arange(0.0001, 0.0011, 0.0001)  # 10 values
    # best instance
    best_acc = 0
    best_c = 0
    best_eta = 0
    # grid search
    with open('grid_search_normal.txt', 'w') as f:
        for c in Cs:
            for eta in ETAs:
                acc = get_cross_validation_accuracy(X, Y, c, eta)
                f.write('{:.1f}\t{:.4f}\t{:.4f}\n'.format(c, eta, acc))
                if acc > best_acc:
                    best_acc = acc
                    best_c = c
                    best_eta = eta
    return best_c, best_eta, best_acc


# def plot_grid_search():
#     """
#     Plot the grid search results.
#     """
#     import matplotlib.pyplot as plt

#     # reads data
#     data = np.loadtxt('grid_search_normal.txt')
#     Cs, ETAs, ACCs = data[:, 0], data[:, 1], data[:, 2]
#     ACCs = ACCs.reshape((5, 10))
#     plt.imshow(ACCs)
#     plt.xticks(np.arange(10), ETAs[:10], rotation=45)
#     plt.yticks(np.arange(5), Cs[::10])
#     plt.xlabel('eta')
#     plt.ylabel('c')
#     plt.title('Cross validation accuracy')
#     plt.colorbar()
#     plt.savefig('grid_search_normal.png')


if __name__=='__main__':
    
    features = sys.argv[1]
    labels = sys.argv[2]
    X, Y = preprocess(features, labels)

    # prints results by fixing c and eta
    best_c = .5
    best_eta = .0004
    best_acc = get_cross_validation_accuracy(X, Y, best_c, best_eta)
    print(best_acc)
    print(best_c)
    print(best_eta)

    # prints results by grid search
    # best_c, best_eta, best_acc = grid_search(X, Y)
    # print(best_acc)
    # print(best_c)
    # print(best_eta)
