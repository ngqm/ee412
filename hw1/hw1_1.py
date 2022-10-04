"""
Filename: hw1_1.py
Author: Quang Minh Nguyen

Submission for homework 1, task 1
References: EE412 lecture slides
"""

import re 
import sys
from pyspark import SparkContext, SparkConf


def preprocess(line):
    """
    Return a profile (user, set of user's friends) from a line
    of text data

    Parameters:
        line: str, 
            <user id><tab><ids of friends separated by commas>
    
    Returns:
        profile: (int, list[int]), (user, set of user's friends). If 
            user has no friend, return (user, []).
    """
    
    user, friends = re.split('\t', line)
    if friends=='':
        friends = []
    else:
        friends = [int(friend) for friend in re.split(',', friends)]
    profile = int(user), friends
    return profile


def get_labeled_triples(profile):
    """
    Return elements (frozenset([user, f1, f2]), user) for every 
    possible set of users {f1, f2} with user as common friend
    given a profile (user, set of user's friend).

    Parameters:
        profile: (int, list[int]), (user, set of user's friends)

    Returns:
        labeled_triples: tuple[([int, int, int], int)], 
            (frozenset([user, f1, f2]), user) for every possible 
            set of users {f1, f2} with user as common friend
    """
    user, friends = profile 
    labeled_triples = []
    for i in range(len(friends)):
        f1 = friends[i]
        for j in range(i+1,len(friends)):
            f2 = friends[j]
            labeled_triples.append((frozenset([user, f1, f2]), user))
    return labeled_triples


def get_potential_friends(labeled_triple):
    """
    Return frozenset([f1, f2]) for a labeled_triple
    (frozenset([user, f1, f2]), user)

    Parameters:
        labeled_triple: tuple[([int, int, int], int)], 
            (frozenset([user, f1, f2]), user)
    
    Returns:
        potential_friends: frozenset([int, int]), a frozen
            set containing two potential friends
    """
    triple, label = labeled_triple
    potential_friends = triple - {label}
    return potential_friends


if __name__=='__main__':

    conf = SparkConf()
    conf.set('spark.logConf', 'true')
    # starts a Spark context 
    sc = SparkContext(conf=conf)

    # reads in text data and produces the required output
    results = sc.textFile(sys.argv[1]) \
        .map(preprocess) \
        .filter(lambda profile: len(profile[1])>1) \
        .flatMap(get_labeled_triples) \
        .reduceByKey(lambda a, b: -1) \
        .filter(lambda labeled_triple: labeled_triple[1]!=-1) \
        .map(lambda labeled_triple: (get_potential_friends(labeled_triple), 1)) \
        .reduceByKey(lambda a, b: a+b) \
        .map(lambda item: (*sorted(list(item[0])), item[1])) \
        .takeOrdered(num=10, key=lambda item: (-item[2], item[0], item[1]))

    # stops the Spark context
    sc.stop()

    # prints out results
    for result in results:
        f1, f2, count = result
        print(f1, f2, count, sep='\t')
