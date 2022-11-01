"""
Filename: hw2_2.py
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
    instead of np.nan for empty arrays.

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


def get_utility_matrix(file):
    """
    Given a text file where each line contains
    <user id>,<movie id>,<rating>,<timestamp>
    separated by commas, return a normalised 
    utility matrix where rows are users and 
    columns are movies.

    Parameters:
        file: str, the text file containing 
            the data
    
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
    # reads in the data
    dataset = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2))
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
    # gets the 10 most similar users
    # excludes user themselves, hence [-11:-1]
    most_similar = np.argsort(similarities)[-11:-1]
    return most_similar


def find_most_similar_movies(movie, pool, utility_matrix):
    """
    Given a movie index, return a list of the 10 most 
    similar movies using cosine similarity.

    Parameters:
        movie: int, the movie index
        pool: ndarray, a list of movie ids that we want
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
    most_similar = pool[np.argsort(similarities)[-10:]]
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
        pool: ndarray, a list of movie ids that we want
            to find the most similar movies from. Required 
            only if method is 'item'. Default is None.
    
    Returns:
        average_rating: float, the predicted rating
    """
    if method == 'user':
        most_similar = find_most_similar_users(user, utility_matrix)
        # find the average rating of the movie among these 10 users
        col = unnormalised_utility[most_similar, movie]
        average_rating = mean(col)  # np.mean(col[col != 0])
    elif method == 'item':
        most_similar = find_most_similar_movies(movie, pool, utility_matrix)
        # find the average rating of the user among these 10 movies
        row = unnormalised_utility[user, most_similar]
        average_rating = mean(row)  # np.mean(row[row != 0])
    return average_rating


if __name__=='__main__':
    pass 