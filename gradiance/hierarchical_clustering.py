from scipy.cluster.hierarchy import dendrogram, single, complete, average, centroid
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


data = [[0,0], [10,10], [21,21], [33,33], [5,27], [28,6]]
labels = ['A', 'B', 'C', 'D', 'E', 'F']

# data = [[i**2] for i in range(1,9)]
# labels = [str(i**2) for i in range(1,9)]


def make_dendrogram(data, method):
    """
    Perform hierarchical clustering on data using
    the specified method, i.e., single, complete, 
    average, or centroid, and plot a dendrogram.

    Parameters:
        data: list of lists of floats
        method: string, one of 'single', 'complete', 
            'average', or 'centroid'

    Returns: None
    """

    y = pdist(data)
    print(y)
    if method == 'single':
        Z = single(y)
    elif method == 'complete':
        Z = complete(y)
    elif method == 'average':
        Z = average(y)
    elif method == 'centroid':
        Z = centroid(y)
    else:
        print('Invalid method')

    dendrogram(Z, labels=labels)


if __name__ == '__main__':

    # from pprint import pprint 
    # import numpy as np 
    # distances = []
    # for i, point1 in enumerate(data):
    #     for j, point2 in enumerate(data[i+1:]):
    #         distances.append((labels[i], labels[j+i+1], pdist([point1, point2])))
    # distances.sort(key=lambda x: x[2])
    # pprint(distances)
        
    plt.figure()
    plt.subplot(221)
    plt.title('Single Linkage')
    make_dendrogram(data, 'single')
    plt.subplot(222)
    plt.title('Complete Linkage')
    make_dendrogram(data, 'complete')
    plt.subplot(223)
    plt.title('Average Linkage')
    make_dendrogram(data, 'average')
    plt.subplot(224)
    plt.title('Centroid')
    make_dendrogram(data, 'centroid')
    plt.show()

    # group1 = [[0,0], [1,0], [0,1], [1,1]]
    # group2 = [[2,0], [3,0], [2,1], [3,1]]

    # print the average distance between any pair of points, 
    # each pair contains one from group1 and one from group2
    # total = 0 
    # for point1 in group1:
    #     for point2 in group2:
    #         total += pdist([point1, point2])
    # print(total / (len(group1) * len(group2)))

    # make_dendrogram(data, 'centroid')
    # plt.show()