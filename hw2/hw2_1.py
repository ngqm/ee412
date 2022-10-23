"""
Filename: hw2_1.py
Author: Quang Minh Nguyen

Python source code for homework 2, task 1
References: Mining of Massive Datasets, Chapter 7
"""

import sys
import numpy as np
from pyspark import SparkContext, SparkConf


def d(x, y, dataset):
    """Euclidean distance between two vectors at indices 
    x and y in the dataset.
    
    Parameters:
        x: int, index of the first vector
        y: int, index of the second vector
        dataset: list[ndarray], a list of vectors
    
    Returns:
        distance: float, the Euclidean distance between 
            the two vectors
    """
    distance = np.linalg.norm(dataset[x] - dataset[y])
    return distance


def general_d(x, S, dataset):
    """
    Return the minimum distance between the vector at index x 
    and any vector with index in S in the dataset.

    Parameters:
        x: int, index of the vector
        S: set, a set of indices of vectors in the dataset
        dataset: list[ndarray], a list of vectors
    
    Returns:
        distance: float, the minimum distance between the vector
            at index x and any vector with index in S in the dataset
    """
    distance = np.min([d(x, y, dataset) for y in S])
    return distance


def get_initial_centroids(first_point, k, dataset):
    """
    Given the index of the first centroid, return a set of 
    k indices of the initial centroids.

    Parameters:
        first_point: int, index of the first centroid
        k: int, number of centroids
        dataset: list[ndarray], a list of vectors
    
    Returns:
        centroids: set[int], a set of indices of the initial centroids
    """

    centroids = {first_point}
    all_points = range(len(dataset))
    while len(centroids) < k:
        distances = [general_d(x, centroids, dataset) for x in all_points]
        next_centroid = np.argmax(distances)
        if next_centroid not in centroids:
            centroids.add(next_centroid)
    return centroids


def find_cluster(x, centroids, dataset):
    """
    Given the index x of a vector in the dataset, return the 
    index of the closest centroid.

    Parameters:
        x: int, index of the vector
        centroids: set[int], a set of indices of the centroids
        dataset: list[ndarray], a list of vectors
    
    Returns:
        index: int, index of the closest centroid in 
            centroids
    """
    distances = [d(x, y, dataset) for y in centroids]
    index = np.argmin(distances)
    return index


def get_diameter(cluster, dataset):
    """
    Given a list of indices of vectors in a cluster
    in a dataset, return the diameter of that cluster.

    Parameters:
        cluster: list[int], a list of indices of vectors in a cluster
        dataset: list[ndarray], a list of vectors
    
    Returns:
        diameter: float, the diameter of the cluster
    """
    diameter = np.max([d(x, y, dataset) for i, x in enumerate(cluster[:-1])
        for y in cluster[i+1:]]) if len(cluster) > 1 else 0
    return diameter


def get_average_diameter(k_value, dataset):
    """
    Perform clustering on dataset using given k_value 
    and return the average diameter of the clusters.

    Parameters:
        k_value: int, number of clusters
        dataset: list[ndarray], a list of vectors
    
    Returns:
        average_diameter: float, the average diameter of the clusters
    """

    # the first point in the dataset is the first centroid
    centroids = get_initial_centroids(0, k_value, dataset)

    # creates a Spark context
    conf = SparkConf()
    conf.set('spark.logConf', 'true') 
    sc = SparkContext(conf=conf)

    # reads in the dataset and produces the required output
    result = sc.parallelize(range(len(dataset))) \
        .map(lambda x: (find_cluster(x, centroids, dataset), x)) \
        .groupByKey() \
        .map(lambda x: [e for e in x[1]]) \
        .map(lambda x: get_diameter(x, dataset)) \
        .reduce(lambda x, y: x + y)
    average_diameter = result / k_value

    return average_diameter


if __name__== '__main__':

    # reads the dataset
    with open(sys.argv[1], 'r') as f:
        dataset = [np.array([float(x) for x in line.split()]) for line in f]
    # reads k 
    k_value = int(sys.argv[2])

    # performs clustering and gets the average diameter of the clusters
    average_diameter = get_average_diameter(k_value, dataset)
    
    # prints the result
    print(average_diameter)
    