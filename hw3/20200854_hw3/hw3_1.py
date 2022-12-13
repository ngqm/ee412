"""
Filename: hw3_1.py
Author: Quang Minh Nguyen

Submission for homework 3, task 1b
References: MMDS book, chapters 2 and 5
"""

import sys
from pyspark import SparkContext, SparkConf


def get_matrix_elements(ego_net):
    """
    Given an ego network around a node, in the form
    (node, [neighbour1, neighbour2, ...]),
    return a list of tuples of the form
    (neighbour, node, 1/deg) for each neighbour.

    Parameters:
        ego_net: a tuple of the form (node, [neighbour1, neighbour2, ...])

    Returns:
        A list of tuples of the form (neighbour, node, 1/deg) for each neighbour.
    """
    node = ego_net[0]
    neighbours = ego_net[1]
    deg = len(neighbours)
    return [(neighbour, node, 1 / deg) for neighbour in neighbours]


def multiply(M, v):
    """
    Given a matrix M and a vector v in their RDD form, 
    return the product Mv.

    Parameters:
        M: RDD, each element is of the form (i, j, M[i,j])
        v: RDD, each element is of the form (j, v[j])
    
    Returns:
        Mv: RDD, each element is of the form (i, sum(M[i,j]v[j]))
    """
    # after join(v): e = (j, ((i, j, M[i, j]), v[j]))
    # after map: e = (i, M[i, j]v[j])
    Mv = M.map(lambda e: (e[1], e)) \
        .join(v).map(lambda e: (e[1][0][0], e[1][0][2] * e[1][1])) \
        .reduceByKey(lambda x, y: x + y)
    return Mv


if __name__=='__main__':

    # constants
    N_ITERS = 50 # number of iterations for pagerank
    N = 1000 # number of pages
    BETA = 0.8

    conf = SparkConf()
    conf.set('spark.logConf', 'true')
    # starts a Spark context 
    sc = SparkContext(conf=conf)

    # gets the edge_list from sys.argv[1]
    edge_list = sc.textFile(sys.argv[1]) \
        .map(lambda line: tuple([int(node) - 1 for node in line.split()])) \
        .distinct()
    
    # gets elements of the transition matrix
    M = edge_list.groupByKey().flatMap(get_matrix_elements)
    M.cache()

    # initialises the pagerank vector to have all entries being 1 / N
    v = sc.parallelize([(i, 1 / N) for i in range(N)])
    
    # computes the terminal pagerank vector
    for _ in range(N_ITERS):
        v = multiply(M, v).map(lambda e: (e[0], BETA * e[1] + (1 - BETA) / N))
        v = sc.parallelize(v.collect())
    
    # collects the top 10 pages
    v = v.takeOrdered(10, key=lambda e: -e[1])

    # prints the top 10 pages with 5 decimal digits 
    [print(f'{line[0]+1}\t{line[1]:.5f}') for line in v]
