"""
Filename: hw1_2.py
Author: Quang Minh Nguyen

Submission for homework 1, task 2
References: EE412 lecture slides
"""

import sys
import numpy as np


def preprocess(file):
    """
    Given a text file containing browsing sessions,
    return a list of lists of items in each session

    Parameters:
        file: str, path to the text file
    
    Returns:
        sessions: list[list[str]], list of lists of items in each session
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    sessions = [list(set(line.split())) for line in lines]
    return sessions 


def get_item_counts(sessions):
    """
    Given a list of list of items in each session,
    return a tuple (item_ids, item_counts) where
    item_ids is a list of item ids and item_counts
    is a list of counts of items with the same index
    in item_ids.

    Parameters:
        sessions: list[list[str]], list of lists of items in each session
    
    Returns:
        item_ids: list[str], list of item ids
        item_counts: list[int], list of counts of items with the same index
            in item_ids
    """
    item_ids = []
    item_counts = []
    for session in sessions:
        for item in session:
            if item in item_ids:
                item_counts[item_ids.index(item)] += 1
            else:
                item_ids.append(item)
                item_counts.append(1)
    return item_ids, item_counts


def get_frequent_items_table(item_counts, support):
    """
    Given a list of counts of items and a support threshold, 
    find all frequent items and return a list where 
    the value at the index of a non-frequent item is 0
    and the value at the index of a frequent item is a 
    number from 1 to the number of frequent items.
    
    Paramters:
        item_counts: list[int], list of counts of items
        support: int, support threshold
    
    Returns:
        frequent_item_counts: int, number of frequent items
        frequent_items_table: list[int], list where the value at the index
            of a non-frequent item is 0 and the value at the index of a 
            frequent item is a number from 1 to the number of frequent items
    """
    frequent_items_table = []
    frequent_item_count = 0
    for count in item_counts:
        if count >= support:
            frequent_item_count += 1
            frequent_items_table.append(frequent_item_count)
        else:
            frequent_items_table.append(0)
    return frequent_item_count, frequent_items_table


def get_triangular_matrix(sessions, item_ids, frequent_item_count, frequent_items_table):
    """
    Given a list of lists of items in each session, a frequent item 
    count, and a frequent items table, return a list triangular_matrix 
    where the count for the pair {i,j}, with 1 <= i < j <= m is stored at
    triangular_matrix[k-1], with k = (i-1)(m-i/2) + j - i

    Parameters:
        sessions: list[list[str]], list of lists of items in each session
        item_ids: list[str], list of item ids
        frequent_item_count: int, number of frequent items
        frequent_items_table: list[int], list where the value at the index
            of a non-frequent item is 0 and the value at the index of a
            frequent item is a number from 1 to the number of frequent items
    
    Returns:
        triangular_matrix: list[int], list where the count for the pair {i,j},
            with 1 <= i < j <= m is stored at triangular_matrix[k], with 
            k = (i-1)(m-i/2) + j - i
        from_k_to_ij: dict[int:tuple(int, int)], dictionary where the key is
            the triangular matrix index k and the value is the pair (i,j)
    """
    m = frequent_item_count
    triangular_matrix = [0] * int(m * (m - 1) / 2)
    from_k_to_ij = {}
    for session in sessions:
        frequent_items = sorted([frequent_items_table[item_ids.index(item)]
            for item in session if frequent_items_table[item_ids.index(item)] != 0])
        for i, item_i in enumerate(frequent_items[:-1]):
            for item_j in frequent_items[i+1:]:
                k = int((item_i - 1) * (m - item_i/2)) + item_j - item_i
                if k not in from_k_to_ij:
                    from_k_to_ij[k] = (item_i, item_j)
                triangular_matrix[k-1] += 1
    return triangular_matrix, from_k_to_ij


def get_most_frequent_pairs(triangular_matrix, frequent_items_table, from_k_to_ij, item_ids):
    """Given triangular matrix, return the 10 most frequent pairs.

    Paramters:
        triangular_matrix: list[int], list where the count for the pair {i,j},
            with 1 <= i < j <= m is stored at index k-1, with 
            k = (i-1)(m-i/2) + j - i
        frequent_items_table: list[int], list where the value at the index
            of a non-frequent item is 0 and the value at the index of a
            frequent item is a number from 1 to the number of frequent items
        from_k_to_ij: dict[int:tuple(int, int)], dictionary where the key is
            the triangular matrix index k and the value is the pair (i,j)
        item_ids: list[str], list of item ids
    
    Returns:
        most_frequent_pairs: list[tuple(int, int), int], list of 10 most 
            frequent pairs and their respective counts
    """
    most_frequent_pairs = []
    for k in np.argsort(triangular_matrix)[-10:][::-1]:
        i, j = from_k_to_ij[k+1]
        id_i = item_ids[frequent_items_table.index(i)]
        id_j = item_ids[frequent_items_table.index(j)]
        most_frequent_pairs.append(((id_i, id_j), triangular_matrix[k]))
    return most_frequent_pairs


if __name__=='__main__':

    SUPPORT = 200

    sessions = preprocess(sys.argv[1])
    item_ids, item_counts = get_item_counts(sessions)
    frequent_item_count, frequent_items_table = get_frequent_items_table(item_counts, SUPPORT)
    triangular_matrix, from_k_to_ij = get_triangular_matrix(sessions, item_ids, frequent_item_count, frequent_items_table)
    frequent_pair_count = len([count for count in triangular_matrix if count >= SUPPORT])
    most_frequent_pairs = get_most_frequent_pairs(triangular_matrix, frequent_items_table, from_k_to_ij, item_ids)
    
    print(frequent_item_count)
    print(frequent_pair_count)
    for pair, count in most_frequent_pairs:
        item_1, item_2 = pair
        print('{}\t{}\t{}'.format(item_1, item_2, count))