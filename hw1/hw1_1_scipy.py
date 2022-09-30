"""
Filename: hw1_1_scipy.py
Author: Quang Minh Nguyen

Alternative algorithm for homework 1, problem 1 using 
    scipy sparse matrices
References: EE412 lecture slides
"""

import re
from scipy.sparse import coo_matrix, diags
import sys


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
    
    user, friends = re.split(r'\t', line)
    if friends=='':
        friends = []
    else:
        friends = [int(friend) for friend in re.split(',', friends)]
    profile = int(user), friends
    return profile


if __name__=='__main__':
    
    # reads in entries of sparse adjacency matrix
    i = []
    j = []
    data = []
    with open(sys.argv[1]) as file:
        for line in file:
            line = line.rstrip('\n')
            user, friends = preprocess(line)
            for friend in friends:
                i.append(user)
                j.append(friend)
                data.append(1)
    
    # creates adjacency matrix
    adj = coo_matrix((data, (i,j))).tocsr()
    # calculates number of common friends
    common_friends = adj@adj
    common_friends -= diags(common_friends.diagonal())

    # gets the pairs of users with the most count of 
    # common friends, provided that they are not friends
    row, col = common_friends.nonzero()
    results = [(*sorted([row[index], col[index]]), common_friends.data[index]) 
        for index in range(len(row)) if adj[row[index], col[index]]==0]
    results = sorted(results, key=lambda x: (-x[2], x[0], x[1]))[:20:2]

    # prints out the result
    for result in results:
        print(result[0], result[1], int(result[2]), sep='\t')

