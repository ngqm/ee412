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
C = .1 # regularisation parameter 
ETA = 0.002 # learning rate
N_ITERS = 1000 # number of iterations
TOL = 1e-4 # tolerance for convergence


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
    W_cur = 0.01*np.random.rand(X.shape[1])
    W = W_cur
    # W = np.zeros(X.shape[1])
    # for _ in range(N_ITERS):
    print('Training...')
    for _ in range(N_ITERS):
        # computes gradient
        grad = np.zeros(X.shape[1])
        for i in range(len(X)):
            if Y[i]*(np.dot(W, X[i])) < 1:
                grad -= Y[i]*X[i]
        grad = W + c*grad
        # update weights
        W -= eta*grad
        # gets the train set accuracy
        predictions = predict(W, X)
        acc = accuracy(predictions, Y)
        print('ACC = {}'.format(acc))
    return W


def predict(W, X):
    """
    Predict labels for data points in X using
    the weights W.

    Parameters:
        W: ndarray, the weights of the SVM
        X: ndarray, each row is a data point
            augmented by a 1
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

    # shuffles the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # splits the data into 10 folds
    X_folds = np.split(X, 10)
    Y_folds = np.split(Y, 10)

    # computes the cross validation accuracy
    acc = 0
    for i in range(10):
        # gets the training data
        X_train = np.concatenate([X_folds[j] for j in range(10) if j != i])
        Y_train = np.concatenate([Y_folds[j] for j in range(10) if j != i])
        # gets the validation data
        X_val = X_folds[i]
        Y_val = Y_folds[i]
        # trains the SVM and get weights
        W = train(X_train, Y_train, c, eta)
        # computes the accuracy
        predictions = predict(W, X_val)
        acc += accuracy(predictions, Y_val)
    acc /= 10
    return acc


def grid_search():
    """
    Perform grid search to find the best hyperparameters
    """
    
    # hyperparameters
    Cs = np.arange(0.1, 1.1, 0.1)
    ETAs = np.arange(0.0001, 0.0011, 0.0001)
    # best instance
    best_acc = 0
    best_c = 0
    best_eta = 0
    # performs grid search
    for c in Cs:
        for eta in ETAs:
            acc = get_cross_validation_accuracy(X, Y, c, eta)
            print('C = {}, ETA = {}, ACC = {}'.format(c, eta, acc))
            if acc > best_acc:
                best_acc = acc
                best_c = c
                best_eta = eta
    print('Best C = {}, Best ETA = {}, Best ACC = {}'.format(best_c, best_eta, best_acc))


if __name__=='__main__':
    
    features = sys.argv[1]
    labels = sys.argv[2]
    X, Y = preprocess(features, labels)
    # X, Y = X[:500], Y[:500]
    error = get_cross_validation_accuracy(X, Y, 1, 0.001)
    print(error)
    # grid_search()


