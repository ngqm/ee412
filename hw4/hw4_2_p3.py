"""
Filename: hw4_2_p3.py
Author:   Quang Minh Nguyen

Submission for Homework 4, Problem 2, using Python 3.
References: MMDS book, Section 4.6
"""

# math operations
import numpy
import math
# file IO
import csv
import os
import sys


class Bucket:
    """
    A class to represent a bucket, with attributes
        size: int, the number of ones in the bucket
        timestamp: int, the timestamp of the leftmost element in the bucket
        next: Bucket, the next bucket
        prev: Bucket, the previous bucket
    """

    def __init__(self, size, timestamp):
        self.size = size
        self.timestamp = timestamp
        self.next = None
        self.prev = None


class BucketList:
    """
    A doubly linked list of buckets, with attributes
        head: Bucket, the head of the list
        tail: Bucket, the tail of the list
        size: int, the number of buckets in the list
    """

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def rebalance(self):
        """
        Traverse from the tail to the head. If there are three 
        buckets with the same size, merge the middle and the 
        left ones.
        """

        # traverses from the tail to the head
        bucket = self.tail
        while bucket is not None:
            # if there are three buckets with the same size
            if (bucket.prev is not None and
                bucket.prev.prev is not None and
                bucket.size == bucket.prev.size and
                bucket.size == bucket.prev.prev.size):
                # merges the middle and the left ones by
                # adding the sizes and use the most recent 
                # timestamp
                bucket.prev.prev.size += bucket.prev.size
                bucket.prev.prev.timestamp = bucket.prev.timestamp
                # removes the middle bucket
                bucket.prev.prev.next = bucket
                bucket.prev = bucket.prev.prev
                # decrements the size of the list
                self.size -= 1
            bucket = bucket.prev
    
    def __str__(self):
        """
        Returns a string representation of the list.
        """
        # traverses from the head to the tail
        bucket = self.head
        result = ""
        while bucket is not None:
            result += str(bucket.size) + " "
            bucket = bucket.next
        return result


def get_buckets(stream):
    """
    Read a stream and return a size array
    and a timestamp array with the same size
    as the number of buckets.

    Parameters:
        stream: str, the path to a stream file, 
            where each line is an element
    
    Returns:
        buckets: BucketList, a doubly linked list of buckets
        stream_size: int, the number of elements in the stream
    """
    # initialises the bucket list
    buckets = BucketList()
    # populate the bucket list
    with open(stream, 'r') as f:
        for i, bit in enumerate(f):
            if bit == '1\n':
                # creates a new bucket
                bucket = Bucket(1, i)
                # if there is no bucket yet
                if buckets.size == 0:
                    buckets.head = bucket
                    buckets.tail = bucket
                    buckets.size = 1
                else:
                    buckets.tail.next = bucket
                    bucket.prev = buckets.tail
                    buckets.tail = bucket
                    buckets.size += 1
                    # rebalances the bucket list
                    buckets.rebalance()
        stream_size = i + 1
    return buckets, stream_size


def query(buckets, stream_size, k_array):
    """
    Estimate the number of 1s among the last
    k bits of the stream for each k in k_array.

    Parameters:
        buckets: BucketList, a doubly linked list of buckets
        stream_size: int, the number of elements in the stream
        k_array: list, a sorted list of k values
    
    Returns:
        estimates: list[float], a list of estimated number of 1s
    """
    k = k_array.pop(0)
    size_sum = 0
    estimates = []
    # traverses from the tail to the head
    bucket = buckets.tail
    while bucket is not None:
        # checks whether some of the last k bits are in the bucket
        if (bucket.timestamp >= stream_size - k and 
            (bucket.prev is None or
            bucket.prev.timestamp < stream_size - k)):
            # makes estimation
            estimate = size_sum + bucket.size/2
            estimates.append(estimate)
            # proceed with the next k
            try:
                # there is still some k left
                next_k = k_array.pop(0)
                # k and next_k might be the same
                while k == next_k:
                    estimates.append(estimate)
                    next_k = k_array.pop(0)
                k = next_k
            except:
                # if there is no more k
                break
        # accumulates size_sum
        size_sum += bucket.size
        bucket = bucket.prev
    return estimates


def get_list_from_indices_and_values(indices, values):
    """
    Given a list containing indices and a list containing values,
    return a list containing values at the indices.

    Parameters:
        indices: list[int], a list of indices
        values: list[float], a list of values
    
    Returns:
        result: list[float], a list of values at the indices
    """
    result = [0]*len(indices)
    for index, value in zip(indices, values):
        result[index] = value
    return result


if __name__=='__main__':
    
    # reads in parameters
    stream = sys.argv[1]
    k_array = [int(k) for k in sys.argv[2:]]
    sorted_indices, sorted_k_array = zip(*sorted(enumerate(k_array), key=lambda x: x[1]))
    sorted_indices = list(sorted_indices)
    sorted_k_array = list(sorted_k_array)

    # computes the queries
    buckets, stream_size = get_buckets(stream)
    unarranged_estimates = query(buckets, stream_size, sorted_k_array)
    estimates = get_list_from_indices_and_values(sorted_indices, unarranged_estimates)
    
    # prints the estimates in the required format
    for estimate in estimates:
        print(estimate)