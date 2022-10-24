import sys
import numpy as np
import matplotlib.pyplot as plt


def generate_plot_data(file, dataset):
    """
    Given a dataset, write the value of 
    k = 1,2,4,... and the corresponding 
    average diameter to each line of file.

    Parameters:
        file: str, the path to the file to write
        dataset: list[ndarray], a list of vectors
    """
    # only this part is dependent on pyspark
    from hw2_1 import get_average_diameter 

    n = len(dataset)
    upperbound = int(np.log2(n))
    # for k in np.logspace(start=0, stop=upperbound, num=upperbound+1, base=2):
    for k in range(4, 9):
        with open(file, 'a') as f:    
            print(f"Working with k={k}")
            average_diameter = get_average_diameter(int(k), dataset)
            f.write(f'{int(k)} {average_diameter}\n')


def plot(file):
    """
    Given a file containing the value of 
    k = 1,2,4,... and the corresponding 
    average diameter, plot the average diameter 
    against k.

    Parameters:
        file: str, the path to the file to read
    """
    with open(file, 'r') as f:
        data = [line.split() for line in f]
    k = [int(e[0]) for e in data]
    average_diameter = [float(e[1]) for e in data]
    plt.plot(k, average_diameter)
    # plt.xscale('log', base=2)
    plt.xticks(range(4,9))
    plt.xlabel('$k$')
    plt.ylabel('Average Diameter')
    plt.title('Average Diameter against Number of Cluster $k=4,5,6,7,8$')
    plt.show()


if __name__=='__main__':

    FILE = 'hw2_1_plot_detailed.txt'
    # reads the dataset
    # with open(sys.argv[1], 'r') as f:
    #     dataset = [np.array([float(x) for x in line.split()]) for line in f]
    # generate_plot_data(FILE, dataset)

    plot(FILE)