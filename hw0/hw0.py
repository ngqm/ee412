"""
Filename: hw0.py
Author: Quang Minh Nguyen

Submission for homework 0.
References: EE412 lecture slides
"""

import re 
import sys
from pyspark import SparkContext, SparkConf


if __name__=='__main__':

    conf = SparkConf()
    conf.set('spark.logConf', 'true')
    # starts a Spark context 
    sc = SparkContext(conf=conf)

    # reads an input file and stores (letter, count) pairs in a list 'results'
    results = sc.textFile(sys.argv[1]) \
        .flatMap(lambda line: re.split(r'[^\w]+', line)) \
        .filter(lambda word: word != '' and word[0].isalpha()) \
        .map(lambda word: word.lower()) \
        .distinct() \
        .map(lambda word: (word[0], 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .sortByKey() \
        .collect()
    # stops the Spark context
    sc.stop()

    # prints the occurences of each letter in the alphabet
    index = 0
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        if index < len(results) and results[index][0] == letter:
            print(letter, results[index][1], sep='\t')
            index += 1
        else:
            print(letter, 0, sep='\t')