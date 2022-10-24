"""
Filename: hw2_2.py
Author: Quang Minh Nguyen

Python source code for homework 2, task 2
References: EE412 lecture slides
"""

import sys
sys.stdout = open('hw2_2.txt', 'w')
import numpy as np


# EXERCISE 11.1.7
print("Exercise 11.1.7\n")


def get_eigenpairs(A, k, epsilon):
    """
    Returns a list containing the first k eigenpairs of A using 
    power iteration until the difference between two consecutive
    vectors is less than epsilon. Also prints the eigenvalues and
    eigenvectors.

    Parameters:
        A: ndarray, a square matrix
        k: int, number of eigenpairs to return
        epsilon: float, threshold for convergence
    
    Returns:
        eigenpairs: list[(float, ndarray)], a list containing 
            the first k eigenpairs of A
    """
    # gets the initial vector
    x = np.array([1, 1, 1])
    # sequentially computes all eigenpairs of A using power iteration
    eigenpairs = []
    for i in range(k):
        # computes the eigenvector
        current_x = x
        while True:
            next_x = A @ current_x
            next_x = next_x / np.linalg.norm(next_x)
            if np.linalg.norm(current_x - next_x) < epsilon:
                break
            current_x = next_x
        eigenvector = next_x
        # computes the eigenvalue   
        eigenvalue = eigenvector @ A @ eigenvector
        eigenpairs.append((eigenvalue, eigenvector))
        print(f"Eigenvalue {i+1}: {eigenvalue}")
        print(f"Eigenvector {i+1}: {eigenvector}")
        if i < k - 1:
            # prepares to find the next eigenpair
            A = A - eigenvalue * np.outer(eigenvector, eigenvector)
            print(f'The newly constructed matrix is \n{A}\n')
    return eigenpairs


# constants
EPSILON = 1e-8
K = 3

# defines the given matrix
B = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6]])
# computes and prints the eigenpairs
eigenpairs = get_eigenpairs(B, K, EPSILON)


print('-----------------------------------------------')

# EXERCISE 11.3.1
print("Exercise 11.3.1\n")

# constants
EPSILON = 1e-8

# defines the given matrix
M = np.array([
    [1, 2, 3], 
    [3, 4, 5],
    [5, 4, 3],
    [0, 2, 4],
    [1, 3, 5]
])

# part (a)
print('(a)')
MTM = M.T @ M
MMT = M @ M.T
print(f'M^T M =\n{MTM},\nM M^T =\n{MMT}\n')

# part (b)
print('(b)')
print('Eigenpairs of M^T M:\n')
eigenvalues, eigenvectors = np.linalg.eig(MTM)
W = []
V = []
for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
    if eigenvalue > EPSILON:
        W.append(eigenvalue)
        V.append(eigenvector)
        print(f'Eigenvalue: {eigenvalue}')
        print(f'Eigenvector: {eigenvector}\n')
V = np.array(V).T

print('Eigenpairs of M M^T:\n')
eigenvalues, eigenvectors = np.linalg.eig(MMT)
W = []
U = []
for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
    if eigenvalue > EPSILON:
        W.append(eigenvalue)
        U.append(eigenvector)
        print(f'Eigenvalue: {eigenvalue}')
        print(f'Eigenvector: {eigenvector}\n')
U = np.array(U).T

# part (c)
print('(c)')
print('The (compact) SVD for the original matrix M:\n')
print(f'U =\n{U},\nSigma =\n{np.sqrt(np.diag(W))},\nV^T =\n{-V.T}\n')

# part (d)
print('(d)')
# keeps only the first singular value 
U_1d = U[:, 0]
W_1d = np.sqrt(W[0])
V_1d = -V[:, 0]
# computes the approximation
M_1d = W_1d * np.outer(U_1d, V_1d)
print(f'The one-dimensional approximation is\n{M_1d}\n')

# part (e)
print('(e)')
print('The ratio of retained energy is\n', W[0]/(W[0] + W[1]))