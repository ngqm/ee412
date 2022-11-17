"""
Filename: hw3_1a.py
Author: Quang Minh Nguyen

Python source code for homework 3, task 1a
References: MMDS book
"""

# linear algebra operations


def multiply_mv(M, v):
    """Multiply a matrix M with a column vector v

    Parameters:
        M: list[list[float]], size n x n
        v: list[float], size n
    """
    return [sum([r*c for r, c in zip(row, v)]) for row in M]


def add_vv(u, v):
    """Add two vectors u and v
    
    Parameters:
        u: list[float], size n
        v: list[float], size n
    """
    return [x + y for x, y in zip(u, v)]


def multiply_sv(s, v):
    """Multiply a scalar s with a vector v

    Parameters:
        s: float
        v: list[float], size n
    """
    return [s * x for x in v]


def distance(u, v):
    """Calculate the Euclidean distance between two vectors u and v

    Parameters:
        u: list[float], size n
        v: list[float], size n
    """
    return sum([(x - y)**2 for x, y in zip(u, v)])**0.5


# PageRank


def get_pagerank(M, beta, tol):
    """
    Get the pagerank of a matrix M with
    1 - beta being the probability
    of teleporting to a random page and
    tol being the tolerance of convergence.

    Parameters:
        M: list[list[float]], size n x n
        beta: float
        tol: float
    """
    n = len(M)
    v_cur = [1. / n] * n
    while True:
        v = add_vv(multiply_sv(beta, multiply_mv(M, v_cur)) , \
            multiply_sv(1 - beta, [1. / n] * n))
        if distance(v, v_cur) < tol:
            break
        v_cur = v
    return v

def get_topic_sensitive_pagerank(M, beta, topic, tol):
    """
    Get the topic-sensitive pagerank of a matrix M with
    1 - beta being the probability
    of teleporting to a random page,
    topic being the indicator vector for the teleport
    set, and tol being the tolerance of convergence.

    Parameters:
        M: list[list[float]], size n x n
        beta: float
        topic: list[int], size n
        tol: float
    """
    n = len(M)
    S = sum(topic)
    v_cur = [1. / n] * n
    while True:
        v = add_vv(multiply_sv(beta, multiply_mv(M, v_cur)) , \
            multiply_sv((1 - beta)/S, topic))
        if distance(v, v_cur) < tol:
            break
        v_cur = v
    return v


# Exercise 5.1.2

M = [
    [1/3, 1/2, 0],
    [1/3, 0, 1/2],
    [1/3, 1/2, 1/2]]

EPSILON = 1e-8
BETA = .8
print("Exercise 5.1.2\nPageRank:\n", get_pagerank(M=M, beta=BETA, tol=EPSILON))


# Exercise 5.3.1

M = [
    [0, 1/2, 1, 0],
    [1/3, 0, 0, 1/2],
    [1/3, 0, 0, 1/2],
    [1/3, 1/2, 0, 0]]

EPSILON = 1e-8
BETA = .8
TOPIC = [1, 0, 0, 0] # A
print("Exercise 5.3.1, (a)\nTopic-sensitive PageRank for A:\n", 
    get_topic_sensitive_pagerank(M=M, beta=BETA, topic=TOPIC, tol=EPSILON))
TOPIC = [1, 0, 1, 0] # A and C
print("Exercise 5.3.1, (b)\nTopic-sensitive PageRank for A and C:\n",
    get_topic_sensitive_pagerank(M=M, beta=BETA, topic=TOPIC, tol=EPSILON))
