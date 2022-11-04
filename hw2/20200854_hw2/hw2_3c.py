"""
Filename: hw2_3c.py
Author: Quang Minh Nguyen

Python source code for homework 2, task 3c
References: Mining of Massive Datasets, Chapter 9
"""

import sys
import numpy as np
from scipy.sparse import csr_array, diags
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


def mean(x):
    """
    Enhance numpy.mean by returning 0 for empty 
    arrays.

    Parameters:
        x: ndarray, an array
    
    Returns:
        mean: float, the mean of x
    """
    if len(x) == 0:
        return 0
    return np.mean(x)


def norm(x):
    """
    Returns the norm of a vector.

    Parameters:
        x: scipy.sparse.csr_matrix, a vector
            with shape 1xn
    
    Returns:
        norm: float, the norm of x
    """
    if x.nnz == 0:
        return 0
    elif x.shape[0] == 1:
        return np.sqrt(x.dot(x.T).toarray()[0][0])
    else:
        return np.sqrt((x.T).dot(x).toarray()[0][0])


def get_dataset(file):
    """
    Given a file where each line contains 
    <user id>,<movie id>,<rating>,<timestamp>, 
    return an ndarray where each row 
    contains <user id>,<movie id>,<rating>.

    Parameters:
        file: str, the text file containing 
            the data.
    
    Returns:
        dataset: ndarray, each row contains
            <user id>,<movie id>,<rating>
    """
    dataset = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2))
    return dataset


def get_input(file):
    """
    Given a file where each line contains 
    <user id>,<movie id>,,<timestamp>, 
    return an ndarray where each row 
    contains <user id>,<movie id>,nan,<timestamp>.

    Parameters:
        file: str, the text file containing 
            the data.
    
    Returns:
        input_data: ndarray, each row contains
            <user id>,<movie id>,nan,<timestamp>
    """
    input_data = np.genfromtxt(file, delimiter=',', missing_values=0)
    return input_data


def get_utility_matrix(dataset):
    """
    Given either an ndarray where each row contains 
    <user id>,<movie id>,<rating>, 
    return a normalised utility matrix 
    where rows are users and columns are movies.

    Parameters:
        dataset: ndarray, each row contains
            <user id>,<movie id>,<rating>
            Required if file is None.
    
    Returns:
        M: scipy.sparse.csr_array, 
            a normalised utility matrix
        uM: scipy.sparse.csr_array, 
            an unnormalised utility matrix
        users: ndarray, a list of user ids
        movies: ndarray, a list of movie ids
        users_inverse : ndarray, a dictionary where keys
            are user ids and values are row indices
        movies_inverse : ndarray, a dictionary where keys
            are movie ids and values are column indices
    """
    # creates sorted users and movies arrays
    users = np.unique(dataset[:, 0].astype(int))
    movies = np.unique(dataset[:, 1].astype(int))
    # creates inverse mapping for users and movies
    users_inverse = {users[i]: i for i in range(len(users))}
    movies_inverse = {movies[i]: i for i in range(len(movies))}
    # creates the sparse utility matrix
    M = csr_array((dataset[:,2], (dataset[:,0].astype(int), dataset[:,1].astype(int))))
    # removes empty rows and columns
    M = M[M.getnnz(1)>0][:,M.getnnz(0)>0]
    # normalises the utility matrix
    uM = M.copy()
    for i in range(M.shape[0]):
        row = M.getrow(i)
        M.data[M.indptr[i]:M.indptr[i+1]] = \
            row.data - row.data.mean()
    return M, uM, users, movies, users_inverse, movies_inverse


def find_most_similar_users(target_user, M, users, users_inverse):
    """
    Given a user index, return a list of the 10 most 
    similar users using cosine similarity.

    Parameters:
        target_user: int, the target user's id
        M: scipy.sparse.csr_array, a normalised
            utility matrix
        users: ndarray, a list of user ids
        users_inverse : ndarray, a dictionary where keys
            are user ids and values are row indices
  
    Returns:
        most_similar: list, a list of the 10 most 
            similar users' ids
    """
    # gets the row index of the target user
    U = users_inverse[target_user]
    # divides every row by its norm
    for u in range(M.shape[0]):
        row = M.getrow(u)
        norm_row = norm(row) if norm(row) else 1
        M.data[M.indptr[u]:M.indptr[u+1]] = row.data / norm_row
    # replaces nan with 1
    M.data[np.isnan(M.data)] = 1
    # computes the cosine similarity between target user and all users
    similarities = M@(M.getrow(U).T)
    # gets the 10 most similar users excluding target user themselves
    sorted_similarity = sorted(range(M.shape[0]),
        key=lambda u: (-similarities[[u]].toarray()[0][0], u))
    most_similar = users[sorted_similarity[1:11]]
    return most_similar


def find_most_similar_movies(target_movie, M, movies, movies_inverse):
    """
    Given a user index, return a list of the 10 most 
    similar users using cosine similarity.

    Parameters:
        target_movie: int, the target movie's id
        M: scipy.sparse.csr_array, a normalised
            utility matrix
        movies: ndarray, a list of movie ids
        movies_inverse : ndarray, a dictionary where keys
            are movie ids and values are column indices
  
    Returns:
        most_similar: list, a list of the 10 most 
            similar movies' ids
    """
    # gets the row index of the target user
    I = movies_inverse[target_movie]
    # turns M into csc format
    M = M.tocsc()
    # divides every column by its norm
    for i in range(M.shape[1]):
        col = M.getcol(i)
        norm_col = norm(col) if norm(col) else 1
        M.data[M.indptr[i]:M.indptr[i+1]] = col.data / norm_col
    # computes the cosine similarity between target user and all users
    similarities = M.T@(M.getcol(I))
    # gets the 10 most similar users excluding target user themselves
    sorted_similarity = sorted(range(M.shape[0]),
        key=lambda i: (-similarities[[i]].toarray()[0][0], i))
    most_similar = movies[sorted_similarity[1:11]]
    return most_similar


def predict_user_based(target_user, target_movie, M, uM, users, users_inverse, movies_inverse):
    """
    Predict the rating of target_user to target_movie using the user-based
    collaborative filtering algorithm.

    Parameters:
        target_user: int, the target user's id
        target_movie: int, the target movie's id
        M: scipy.sparse.csr_array, a normalised
            utility matrix
        uM: scipy.sparse.csr_array, an unnormalised
            utility matrix
        users: ndarray, a list of user ids
        users_inverse : ndarray, a dictionary where keys
            are user ids and values are row indices
        movies_inverse : ndarray, a dictionary where keys
            are movie ids and values are column indices
  
    Returns:
        prediction: float, the predicted rating
    """
    # gets the row index of the target user and column index of the target movie
    try:
        U = users_inverse[target_user]
        try:
            # both U and I exists
            I = movies_inverse[target_movie]
            # gets the 10 most similar users
            most_similar_users = find_most_similar_users(target_user, M, users, users_inverse)
            # gets the row indices of the 10 most similar users
            most_similar_users = [users_inverse[u] for u in most_similar_users]
            # gets the average rating of the 10 most similar users to the target movie,
            # ignoring those who have not rated the target movie
            ratings = [M.getcol(I).getrow(u).toarray()[0][0] for u in 
                set(most_similar_users) & set(uM.getcol(I).indices)]
            prediction = mean(ratings) + mean(uM.getrow(U).data)
        except:
            # U exists but I does not
            prediction = mean(uM.getrow(U).data)
    except:
        try: 
            # U does not exist but I exists
            I = movies_inverse[target_movie]
            prediction = mean(uM.getcol(I).data)
        except:
            # neither U nor I exists
            prediction = mean(uM.data)

    return prediction


def predict_item_based(target_user, target_movie, M, uM, movies, users_inverse, movies_inverse):
    """
    Predict the rating of target_user to target_movie using the item-based
    collaborative filtering algorithm.

    Parameters:
        target_user: int, the target user's id
        target_movie: int, the target movie's id
        M: scipy.sparse.csr_array, a normalised
            utility matrix
        uM: scipy.sparse.csr_array, an unnormalised
            utility matrix
        movies: ndarray, a list of movie ids
        users_inverse : ndarray, a dictionary where keys
            are user ids and values are row indices
        movies_inverse : ndarray, a dictionary where keys
            are movie ids and values are column indices
  
    Returns:
        prediction: float, the predicted rating
    """
    # gets the row index of the target user and column index of the target movie
    try:
        U = users_inverse[target_user]
        try:
            # both U and I exists
            I = movies_inverse[target_movie]
            # gets the 10 most similar movies
            most_similar_movies = find_most_similar_movies(target_movie, M, movies, movies_inverse)
            # gets the column indices of the 10 most similar movies
            most_similar_movies = [movies_inverse[i] for i in most_similar_movies]
            # gets the average rating of the 10 most similar movies to the target user,
            # ignoring those who have not rated the target movie
            ratings = [M.getrow(U).getcol(i).toarray()[0][0] for i in 
                set(most_similar_movies) & set(uM.getrow(U).indices)]
            prediction = mean(ratings) + mean(uM.getrow(U).data)
        except:
            # U exists but I does not
            prediction = mean(uM.getrow(U).data)
    except:
        try: 
            # U does not exist but I exists
            I = movies_inverse[target_movie]
            prediction = mean(uM.getcol(I).data)
        except:
            # neither U nor I exists
            prediction = mean(uM.data)
    return prediction


def get_svd(M, rank):
    """
    Return the reduced singular value decomposition of M
    with specified rank using scipy.sparse.linalg.svds.

    Parameters:
        M: scipy.sparse.csr_array, a normalised
            utility matrix
        k: int, the rank of the reduced SVD
    
    Returns:
        U_svd: scipy.sparse.csr_array, the left singular vectors
        S_svd: scipy.sparse.csr_array, the singular values
        Vt_svd: scipy.sparse.csr_array, the right singular vectors
    """
    U_svd, S_svd, Vt_svd = svds(M, k=rank)
    S_svd = diags(S_svd)
    return U_svd, S_svd, Vt_svd


def predict_svd(target_user, target_movie, U, S, Vt, uM, users_inverse, movies_inverse):
    """
    Predict the rating of target_user to target_movie using the SVD
    collaborative filtering algorithm.

    Parameters:
        target_user: int, the target user's id
        target_movie: int, the target movie's id
        U: scipy.sparse.csr_array, the left singular vectors
        S: scipy.sparse.csr_array, the singular values
        Vt: scipy.sparse.csr_array, the right singular vectors
        uM: scipy.sparse.csr_array, an unnormalised
            utility matrix
        users: ndarray, a list of user ids
        movies: ndarray, a list of movie ids
        users_inverse : ndarray, a dictionary where keys
            are user ids and values are row indices
        movies_inverse : ndarray, a dictionary where keys
            are movie ids and values are column indices
  
    Returns:
        prediction: float, the predicted rating
    """
    # gets the row index of the target user and column index of the target movie
    try:
        U = users_inverse[target_user]
        try:
            # both U and I exists
            I = movies_inverse[target_movie]
            prediction = 1/2*((U@S)@Vt[:,[I]] + mean(uM.getrow(U).data)) + \
                1/2*((U@S)@Vt[[U],:] + mean(uM.getcol(I).data))
        except:
            # U exists but I does not
            prediction = mean(uM.getrow(U).data)
    except:
        try: 
            # U does not exist but I exists
            I = movies_inverse[target_movie]
            prediction = mean(uM.getcol(I).data)
        except:
            # neither U nor I exists
            prediction = mean(uM.data)
    return prediction


def rmse(predictions, targets):
    """
    Computes the root mean squared error between 
    predictions and targets.

    Parameters:
        predictions: ndarray, a list of predictions
        targets: ndarray, a list of targets
    
    Returns:
        rmse: float, the root mean squared error
    """
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    return rmse


def get_cross_validation_error(dataset, M, uM, method, users, movies, rank=10, k=10):
    """
    Computes the k-fold cross validation error of the 
    dataset.

    Parameters:
        dataset: ndarray, a list of (user, movie, rating)
            tuples
        M: scipy.sparse.csr_array, a normalised
            utility matrix
        uM: scipy.sparse.csr_array, an unnormalised
            utility matrix
        users: ndarray, a list of user ids
        movies: ndarray, a list of movie ids
  
    Returns:
        error: float, the cross validation error
    """
    from time import time
    # splits the dataset into k folds
    np.random.shuffle(dataset)
    folds = np.array_split(dataset, k)
    rmse_sum = 0
    for fold in range(k):
        start = time()
        # gets the training set
        train = np.concatenate([folds[i] for i in range(k) if i != fold])
        # gets the test set
        test = folds[fold]
        # gets the training utility matrix
        M, uM, users, movies, users_inverse, movies_inverse = get_utility_matrix(train)
        # computes the rmse on the test set
        if method == 'user_based':
            # uses user based collaborative filtering
            predictions = [predict_user_based(int(U), int(I), M, uM, movies, users_inverse, movies_inverse) for U, I, _ in test]
        elif method == 'item_based':
            # uses item based collaborative filtering
            predictions = [predict_item_based(int(U), int(I), M, uM, users, movies_inverse, users_inverse) for U, I, _ in test]
        elif method == 'svd':
            # uses SVD collaborative filtering
            U_svd, S_svd, Vt_svd = get_svd(M, rank)
            predictions = [predict_svd(int(U), int(I), U_svd, S_svd, Vt_svd, uM, users_inverse, movies_inverse) for U, I, _ in test]
        targets = [float(r) for _, _, r in test]
        rmse_sum += rmse(np.array(predictions), np.array(targets))
        print('time for one fold: ', time() - start)
    error = rmse_sum / k
    return error


def main(output_file):
    """
    Each row of training_file, sys.argv[1], is in the form
    <user id>,<movie id>,<rating>,<timestamp>;
    each row of input_file, sys.argv[2], is in the form
    <user id>,<movie id>,,<timestamp>. Fill in
    the rating for input_file and save to
    output_file with the format
    <user id>,<movie id>,<rating>,<timestamp>.

    Parameters:
        output_file: str, the path to the output file
    """
    # reads the training data and the input file
    dataset = get_dataset(sys.argv[1])
    input_data = get_input(sys.argv[2])
    # gets the utility matrix
    M, uM, users, movies, users_inverse, movies_inverse = get_utility_matrix(dataset)
    # uses SVD collaborative filtering
    U_svd, S_svd, Vt_svd = get_svd(M, rank=30)
    # computes the predictions
    predictions = [predict_svd(int(U), int(I), U_svd, S_svd, Vt_svd, uM, users_inverse, movies_inverse) for U, I, _, _ in input_data]
    # writes the predictions to the output file
    with open(output_file, 'w') as f:
        for i in range(len(input_data)):
            f.write(str(int(input_data[i][0])) + ',' + str(int(input_data[i][1])) + ',' + str(predictions[i]) + ',' + str(int(input_data[i][3])) + '\n')
            

if __name__=='__main__':
    
    main('output.txt')
