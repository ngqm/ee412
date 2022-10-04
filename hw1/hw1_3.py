"""
Filename: hw1_3.py
Author: Quang Minh Nguyen

Submission for homework 1, task 3
References: EE412 lecture slides, MMDS book
"""


import numpy as np
import sys


def is_prime(n):
    """Check if an integer n is a prime

    Parameters:
        n: int

    Returns:
        is_prime_bool: bool, True if n is a prime and 
            False otherwise
    """
    if n%2 == 0 and n>2:
        return False        
    for i in range(3, int(np.sqrt(n))+1, 2):
        if n % i == 0:
            return False
    return True


def get_least_upperbound_prime(n):
    """
    Given an integer n, return the least prime
    that is greater than or equal to n

    Parameters:
        n: int
    
    Returns:
        p: int, the least prime that is greater than
            or equal to n
    """

    while not is_prime(n):
        n += 1
    return n


def get_random_hash_function(n):
    """
    Given a number n, find c, which is 
    the least prime that is greater than or equal to n,
    then set a, b to be random integers between 0 and c-1

    Parameters:
        n: int

    Returns:
        hash_function: function, a hash function which
            receives x and outputs (a+b*x) mod c
    """
    c = get_least_upperbound_prime(n)
    a = np.random.randint(0, c)
    b = np.random.randint(0, c)
    hash_function = lambda x: (a + b*x) % c
    return hash_function


def preprocess(file):
    """Given a text file where each line contains 
    a document id and the main text separated by a tab, 
    return a list of set of 3-shingles in each document

    Parameters:
        file: str, path to the text file
    
    Returns:
        documents: list[set], list of set of 3-shingles in each document
        document_ids: list[str], list of document ids
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    documents = []
    document_ids = []
    for line in lines:
        first_space = line.find(' ')
        document_id = line[:first_space]
        text = line[first_space+1:]
        text = ''.join([c.lower() for c in text if c.isalpha() or c==' '])
        shingles = set()
        for i in range(len(text)-2):
            shingle = text[i:i+3]
            shingles.add(shingle)
        documents.append(shingles)
        document_ids.append(document_id)
    return documents, document_ids


def get_signature_matrix(documents, n_hash_functions):
    """Given a list of set of 3-shingles in each document
    and a number of hash functions, return a signature matrix

    Parameters:
        documents: list[set], list of set of 3-shingles in each document
        n_hash_functions: int, number of hash functions

    Returns:
        signature_matrix: np.ndarray, signature matrix of shape 
            (n_hash_functions, len(documents))
    """
    n_documents = len(documents)
    all_shingles = list(set().union(*documents))
    n_shingles = len(all_shingles)
    hash_functions = []
    [hash_functions.append(get_random_hash_function(n_shingles)) 
        for _ in range(n_hash_functions)]
    signature_matrix = np.full((n_hash_functions, n_documents), np.inf)
    for r in range(n_shingles):
        hash_values = [hash_function(r) for hash_function in hash_functions]
        for c, document in enumerate(documents):
            if all_shingles[r] in document:
                for i in range(n_hash_functions):
                    signature_matrix[i, c] = min(signature_matrix[i, c], hash_values[i])
    return signature_matrix


def get_candidates(signature_matrix, n_bands, n_rows):
    """
    Given a signature matrix, number of bands, and 
    number of rows per band, return a list of candidate pairs

    Parameters:
        signature_matrix: np.ndarray, signature matrix of shape 
            (n_hash_functions, len(documents))
        n_bands: int, number of bands
        n_rows: int, number of rows per band
        threshold: float, threshold for similarity

    Returns:
        candidates: set[frozenset[int]], set of candidate pairs
    """

    candidates = set()
    _, n_documents = signature_matrix.shape
    for band_id in range(n_bands):
        band = signature_matrix[band_id*n_rows:(band_id+1)*n_rows]
        signatures = band.T
        for i in range(n_documents):
            for j in range(i+1, n_documents):
                if np.array_equal(signatures[i], signatures[j]):
                    candidates.add(frozenset([i, j]))
    return candidates


def jaccard_similarity(document1, document2):
    """Given two sets of 3-shingles, return the Jaccard similarity

    Parameters:
        document1: set, set of 3-shingles
        document2: set, set of 3-shingles

    Returns:
        similarity: float, Jaccard similarity
    """

    intersection = len(document1.intersection(document2))
    union = len(document1.union(document2))
    return intersection/union


def get_similar_pairs(candidates, documents, threshold):
    """
    Given a set of candidate pairs and a list of set of 
    3-shingles in each document, return a list of candidate pairs
    that actually have Jaccard similarity higher than a threshold

    Parameters:
        candidates: set[tuple[int]], set of candidate pairs
        documents: list[set], list of set of 3-shingles in each document
        threshold: float, threshold for similarity

    Returns:
        similar_pairs: list[tuple[int]], list of similar pairs
    """

    similar_pairs = []
    for pair in candidates:
        i, j = pair
        similarity = jaccard_similarity(documents[i], documents[j])
        if similarity >= threshold:
            similar_pairs.append(pair)
    return similar_pairs


if __name__=='__main__':

    # hyperparameters
    B = 6
    R = 20
    THRESHOLD = 0.9

    documents, document_ids = preprocess(sys.argv[1])
    signature_matrix = get_signature_matrix(documents, B*R)
    candidates = get_candidates(signature_matrix, B, R)
    similar_pairs = get_similar_pairs(candidates, documents, THRESHOLD)

    for pair in similar_pairs:
        document1, document2 = pair
        print("{}\t{}".format(document_ids[document1], document_ids[document2]))