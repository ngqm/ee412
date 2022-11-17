"""
Filename: hw3_2_p3.py
Author: Quang Minh Nguyen

Submission for homework 3, task 2b
References: MMDS book, Section 10.7.2
"""

import sys 
import math 
from itertools import combinations
from time import time 


def preprocess(file):
    """
    Read in a file where each line has the format
    <user1> <tab> <user2> <tab> <timestamp> 
    Compute
        (1) A hash table for node degrees
        (2) A hash table for the existence of an edge
        (3) A hash table for node neighbors
    
    Parameters:
        file: str, the file to be read

    Returns:
        deg_dict: dict[int:int], the hash table for node degrees
        edge_dict: dict[(int,int):int], the hash table for the existence of an edge
        edge_set: set[(int, int)], the set of edges
        neighbour_dict: dict[int:set[int]], the hash table for node neighbours
    """
    deg_dict = {}
    edge_dict = {}
    edge_set = set()
    neighbour_dict = {}
    with open(file, 'r') as f:
        for line in f:
            user1, user2, _ = line.strip().split('\t')
            user1, user2 = int(user1), int(user2)
            user1, user2 = min(user1, user2), max(user1, user2)
            if (user1, user2) not in edge_set:
                edge_set.add((user1, user2))
                # updates the degree hash table
                deg_dict[user1] = deg_dict.get(user1, 0) + 1
                deg_dict[user2] = deg_dict.get(user2, 0) + 1
                # updates the edge existence hash table
                edge_dict[(user1, user2)] = 1
                # updates the neighbour hash table
                neighbour_dict[user1] = neighbour_dict.get(user1, set()) | {user2}
                neighbour_dict[user2] = neighbour_dict.get(user2, set()) | {user1}
    return deg_dict, edge_dict, edge_set, neighbour_dict


def sort_nodes_by_deg(deg_dict):
    """
    Sort the nodes by their degrees in ascending order. If 
    two nodes have the same degree, sort them by their ids
    in ascending order.

    Parameters:
        deg_dict: dict[int:int], the hash table for node degrees

    Returns:
        sorted_nodes: list[int], the list of nodes sorted by their degrees
    """
    sorted_nodes = sorted(deg_dict.items(), key=lambda x: -x[1])
    sorted_nodes = [node for node, _ in sorted_nodes]
    return sorted_nodes


def get_heavy_hitters(deg_dict, sorted_nodes, m):
    """
    Get the heavy hitter nodes from the degree hash table
    and the number of edges m.

    Parameters:
        deg_dict: dict[int:int], the hash table for node degrees
        sorted_nodes: list[int], the list of nodes sorted by their degrees
        m: int, the number of edges
    
    Returns:
        heavy_hitters: set[int], the set of heavy hitter nodes
    """
    # real-world graphs have a power law degree distribution,
    # so very few nodes will be heavy hitters,
    # therefore linear search is sufficient
    heavy_hitters = set()
    for node in sorted_nodes:
        if deg_dict[node] > math.sqrt(m):
            heavy_hitters.add(node)
        else:
            break
    return heavy_hitters


def count_heavy_hitter_triangles(heavy_hitters, edge_set):
    """Count the number of heavy hitter triangles.

    Parameters:
        heavy_hitters: set[int], the set of heavy hitter nodes
        edge_set: set[(int, int)], the set of edges
    
    Returns:
        count: int, the count of heavy hitter triangles
    """
    count = sum([1 for n1, n2, n3 in combinations(heavy_hitters, 3) if 
                (n1, n2) in edge_set and 
                (n1, n3) in edge_set and 
                (n2, n3) in edge_set])
    return count


def preceeds(n1, n2, deg_dict):
    """Return True if n1 has lower degree than n2
    or if n1 and n2 have the same degree but n1 < n2;
    return False otherwise.

    Parameters:
        n1: int, the first node
        n2: int, the second node
        deg_dict: dict[int:int], the hash table for node degrees

    Returns:
        prec: bool
    """
    if deg_dict[n1] < deg_dict[n2]:
        return True
    elif deg_dict[n1] == deg_dict[n2] and n1 < n2:
        return True
    return False


def count_other_triangles(deg_dict, edge_dict, edge_set, neighbour_dict, heavy_hitters):
    """Count the number of non-heavy-hitter triangles.
    
    Parameters:
        deg_dict: dict[int:int], the hash table for node degrees
        edge_dict: dict[(int,int):int], the hash table for the existence of an edge
        edge_set: set[(int, int)], the set of edges
        neighbour_dict: dict[int:set[int]], the hash table for node neighbours
        heavy_hitters: set[int], the set of heavy hitter nodes
    
    Returns:
        count: int, the count of non-heavy-hitter triangles
    """
    count = 0 
    for n1, n2 in edge_dict:
        if n1 in heavy_hitters and n2 in heavy_hitters:
            continue
        if not preceeds(n1, n2, deg_dict):
            n1, n2 = n2, n1
        for n3 in neighbour_dict[n1]:
            x, y = min(n2, n3), max(n2, n3)
            if (x, y) in edge_set and preceeds(n2, n3, deg_dict):
                count += 1
    return count


if __name__=='__main__':
    
    # reads in the file
    file = sys.argv[1]
    # preprocessing steps
    deg_dict, edge_dict, edge_set, neighbour_dict = preprocess(file)
    sorted_nodes = sort_nodes_by_deg(deg_dict)
    heavy_hitters = get_heavy_hitters(deg_dict, sorted_nodes, len(edge_set))
    # counts the number of triangles
    count_heavy = count_heavy_hitter_triangles(heavy_hitters, edge_set)
    count_other = count_other_triangles(deg_dict, edge_dict, edge_set, neighbour_dict, heavy_hitters)
    total_count = count_heavy + count_other
    # prints final result
    print(total_count)