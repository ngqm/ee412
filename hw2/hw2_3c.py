"""
Filename: hw2_3c.py
Author: Quang Minh Nguyen

Python source code for homework 2, task 3c
References: Mining of Massive Datasets, Chapter 9
"""

import sys
import numpy as np
from numpy.linalg import norm


def mean(array):
    """
    An enhanced version of numpy.mean that returns 0
    instead of numpy.nan for empty arrays.

    Parameter:
        array: ndarray
    
    Returns:
        m: float, the mean of the array
    """
    if len(array[array != 0]) == 0:
        m = 0
    else:
        m = np.mean(array[array != 0])
    return m


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


def get_utility_matrix(file=None, dataset=None):
    """
    Given either 
    (1) a text file where each line contains
    <user id>,<movie id>,<rating>,<timestamp>
    separated by commas or 
    (2) an ndarray where each row contains 
    <user id>,<movie id>,<rating>, 
    return a normalised utility matrix 
    where rows are users and columns are movies.

    Parameters:
        file: str, the text file containing 
            the data. Required if dataset is None.
        dataset: ndarray, each row contains
            <user id>,<movie id>,<rating>
            Required if file is None.
    
    Returns:
        users, movies, utility_matrix, unnormalised_utility:
            users: ndarray, a list of user ids
            users_inverted: dict, a dictionary mapping
                user ids to user indices
            movies: ndarray, a list of movie ids
            movies_inverted: dict, a dictionary mapping
                movie ids to movie indices
            utility_matrix: ndarray, a normalised
                utility matrix
            unnormalised_utility: ndarray, an unnormalised
                utility matrix
    """
    if dataset is None:
        # reads in the data
        dataset = get_dataset(file)
    # creates users and movies arrays
    users   = np.sort(np.unique(dataset[:, 0])).astype(int)
    users_inverted = {users[i]: i for i in range(len(users))}
    movies  = np.sort(np.unique(dataset[:, 1])).astype(int)
    movies_inverted = {movies[i]: i for i in range(len(movies))}
    # initilises the utility matrix with zeros
    utility_matrix = np.zeros((len(users), len(movies)))
    # fills in the utility matrix
    for row in dataset:
        user    = int(row[0])
        movie   = int(row[1])
        rating  = row[2]
        utility_matrix[users_inverted[user], movies_inverted[movie]] = rating
    unnormalised_utility = utility_matrix.copy()
    # normalises the utility matrix
    for i in range(len(users)):
        row = utility_matrix[i]
        row[row != 0] -= mean(row)
    return users, users_inverted, movies, movies_inverted, \
        utility_matrix, unnormalised_utility


def find_most_similar_users(user, utility_matrix):
    """
    Given a user index, return a list of the 10 most 
    similar users using cosine similarity.

    Parameters:
        user: int, the user index
        utility_matrix: ndarray, a normalised
            utility matrix
    
    Returns:
        most_similar: list, a list of the 10 most 
            similar users' indices
    """
    # gets the normalised ratings of user
    ratings = utility_matrix[user]
    # computes the cosine similarity between user and all users
    if norm(ratings) == 0:
        similarities = np.zeros(len(utility_matrix))
    else:
        unnormed_similarities = utility_matrix@ratings.T / norm(ratings)
        stable_norm = np.array([norm(u) if norm(u) > 0 else 1 
            for u in utility_matrix])
        similarities = unnormed_similarities / stable_norm
    # gets the 10 most similar users excluding user themselves
    most_similar = sorted(range(len(similarities)), key=lambda k: (-similarities[k], k))
    count = 0
    new_most_similar = []
    for u in most_similar:
        if u != user:
            new_most_similar += [u]
            count += 1
            if count == 10:
                break
    most_similar = new_most_similar
    return most_similar


def find_most_similar_movies(movie, pool, utility_matrix):
    """
    Given a movie index, return a list of the 10 most 
    similar movies using cosine similarity.

    Parameters:
        movie: int, the movie index
        pool: ndarray, a list of movie indices that we want
            to find the most similar movies from
        utility_matrix: ndarray, a normalised
            utility matrix
    
    Returns:
        most_similar: list, a list of the 10 most 
            similar movies' indices
    """
    # gets the normalised ratings of movie
    ratings = utility_matrix[:, movie]
    # gets the normalised ratings of the movies we want to find
    # the most similar movies from
    other_ratings = utility_matrix[:, pool]
    # computes the cosine similarity between movie and all movies
    if norm(ratings) == 0:
        similarities = np.zeros(len(pool))
    else:
        unnormed_similarities = other_ratings.T@ratings / norm(ratings)
        stable_norm = np.array([norm(m) if norm(m) > 0 else 1 
            for m in other_ratings.T])
        similarities = unnormed_similarities / stable_norm
    # gets the 10 most similar movies' indices
    # excluding movie itself
    most_similar = sorted(range(len(similarities)), key=lambda k: (-similarities[k], k))
    count = 0
    new_most_similar = []
    for m in most_similar:
        if m != movie:
            new_most_similar += [m]
            count += 1
            if count == 10:
                break
    most_similar = new_most_similar
    most_similar = pool[most_similar]
    return most_similar


def predict(user, movie, method, utility_matrix, unnormalised_utility, pool=None):
    """
    Predict the rating of a user to a movie.

    Parameters:
        user: int, the user index
        movie: int, the movie index
        method: str, either 'user', meaning user-based,
            or 'item', meaning item-based
        utility_matrix: ndarray, a normalised
            utility matrix
        unnormalised_utility: ndarray, an unnormalised
            utility matrix
        pool: ndarray, a list of movie indices that we want
            to find the most similar movies from. Required 
            only if method is 'item'. Default is None.
    
    Returns:
        average_rating: float, the predicted rating
    """
    if method == 'user':
        if movie != -1 and user != -1:
            most_similar = find_most_similar_users(user, utility_matrix)
            # find the average rating of the movie among these 10 users
            col = unnormalised_utility[most_similar, movie]
            average_rating = mean(col)
        elif movie == -1 and user != -1:
            # find the average rating of the user
            row = unnormalised_utility[user]
            average_rating = mean(row)
        elif movie != -1 and user == -1:
            # find the average rating of the movie
            col = unnormalised_utility[:, movie]
            average_rating = mean(col)
        else:
            # find the average rating of all movies
            average_rating = mean(unnormalised_utility)
    elif method == 'item':
        if movie != -1 and user != -1:
            most_similar = find_most_similar_movies(movie, pool, utility_matrix)
            # find the average rating of the user among these 10 movies
            row = unnormalised_utility[user, most_similar]
            average_rating = mean(row)
        elif movie == -1 and user != -1:
            # find the average rating of the user
            row = unnormalised_utility[user]
            average_rating = mean(row)
        elif movie != -1 and user == -1:
            # find the average rating of the movie
            col = unnormalised_utility[:, movie]
            average_rating = mean(col)
        else:
            # find the average rating of all movies
            average_rating = mean(unnormalised_utility)
    average_rating = round(average_rating * 2) / 2
    return average_rating


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


def get_cross_validate_error(dataset, method, k=20):
    """
    Performs k-fold cross validation on the dataset.
    Returns the average RMSE.

    Parameters:
        dataset: ndarray, the dataset to be split. 
            Each row contains 
            <user id>, <movie id>, <rating>
        method: str, either 'user', meaning user-based,
            or 'item', meaning item-based
        k: int, the number of folds

    Returns:
        rmse: float, the average root mean squared error
    """
    # splits the dataset into k folds
    np.random.shuffle(dataset)
    folds = np.array_split(dataset, k)
    # computes the average rmse
    rmse_sum = 0
    for i in range(k):
        # gets the test set
        test = folds[i]
        # gets the training set
        train = np.concatenate(folds[:i] + folds[i+1:])
        # gets the utility matrix
        users, users_inverted, movies, movies_inverted, \
            utility_matrix, unnormalised_utility = get_utility_matrix(dataset=train)
        # gets the pool of movies to find the most similar movies from
        pool = np.array([i for i in range(len(movies))])
        # makes predictions on the test set
        predictions = []
        targets = []
        for row in test:
            # -1 means the user or movie is not in the training set
            user = users_inverted.get(int(row[0]), -1) 
            movie = movies_inverted.get(int(row[1]), -1)
            target = float(row[2])
            prediction = predict(user, movie, method, utility_matrix, unnormalised_utility, pool)
            predictions += [prediction]
            targets += [target]
        # computes the rmse
        rmse_sum += rmse(np.array(predictions), np.array(targets))
    average_rmse = rmse_sum / k
    return average_rmse


if __name__=='__main__':
    
    # gets the dataset
    dataset = get_dataset(sys.argv[1])
    # performs k-fold cross validation
    rmse_user = get_cross_validate_error(dataset, 'user')
    print('user-based rmse:', rmse_user)
    rmse_item = get_cross_validate_error(dataset, 'item')
    print('item-based rmse:', rmse_item)
