"""
Filename: hw2_3b.py
Author: Quang Minh Nguyen

Python source code for homework 2, task 3b
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


def find_top_movies(user, movies_inverted, method, utility_matrix, unnormalised_utility, pool=None):
    """
    Find the top 5 movies among movies 1 to 1000 for a user with
    the highest predicted rating.

    Parameters:
        user: int, the user index
        movies_inverted: dict, a dictionary that maps movie ids
            to their indices in the utility matrix
        method: str, either 'user', meaning user-based,
            or 'item', meaning item-based
method   
    Returns:
        top_movies: list, a list of the top 5 movies
            for the user
        predicted_ratings: list, a list of the predicted
            ratings for the top 5 movies
    """
    # finds the predicted ratings for all movies
    predicted_ratings = np.zeros(1000)
    for movie_id in range(1, 1001):
        if movie_id in movies_inverted:
            predicted_ratings[movie_id-1] = predict(
                user, movies_inverted[movie_id], method, 
                utility_matrix, unnormalised_utility,
                pool)
    # gets the top 5 movies and their predictions
    # if two movies have the same predicted rating,
    # the one with the lower id is ranked higher
    movies_and_ratings = sorted(
        zip(range(1000), predicted_ratings), 
        key=lambda x: (-x[1], x[0]))[:5]
    top_movies = [m+1 for m, _ in movies_and_ratings]
    predicted_ratings = [r for _, r in movies_and_ratings]
    return top_movies, predicted_ratings


if __name__=='__main__':
    
    # constant
    U = 600

    # extracts the data
    users, users_inverted, movies, movies_inverted, M, uM = \
        get_utility_matrix(sys.argv[1])
    # gets indices of movies with ids outside of range(1, 1001)
    MOVIES = np.array([movies_inverted[i] for i in movies_inverted.keys() 
                        if i > 1000 or i < 1])

    # gets top-5 movies with highest predicted ratings using user-based method
    top5_user, predicted_ratings = find_top_movies(
        users_inverted[U], movies_inverted,
        'user', M, uM)
    for movie, rating in zip(top5_user, predicted_ratings):
        print(f'{movie}\t{rating}')
    # gets top-5 movies with highest predicted ratings using item-based method
    top5_item, predicted_ratings = find_top_movies(
        users_inverted[U], movies_inverted,
        'item', M, uM, MOVIES)
    for movie, rating in zip(top5_item, predicted_ratings):
        print(f'{movie}\t{rating}')
