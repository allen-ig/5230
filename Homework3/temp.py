import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.spatial import *

## you need to change the directory to your local dirs
directory = 'datasets/'
#Users/allen/Desktop/ds5230/Homework3/
dataset1 = np.loadtxt(directory + "dataset1.txt",
                    dtype = {'names': ('x1', 'x2', 'label'), 'formats': (float, float, int)},
                    delimiter="\t")
dataset2 = np.loadtxt(directory + "dataset2.txt",
                   dtype = {'names': ('x1', 'x2', 'label'), 'formats': (float, float, int)},
                   delimiter="\t")
dataset3 = np.loadtxt(directory + "dataset3.txt",
                   dtype = {'names': ('x1', 'x2', 'label'), 'formats': (float, float, int)},
                   delimiter="\t")


def create_io_matrix(d):
    """
    formats the input data in a matrix format (n × d) where d is the number of dimentions and n is the number of data points
    formats the label data in a matrix format (n × 1) where n is the number of data points
    """
    matrix_input = [[x1, x2] for x1, x2, l in d]
    matrix_labels = [l for x1, x2, l in d]

    return (np.array(matrix_input), np.array(matrix_labels))


dataset1_matrix, data1_labels = create_io_matrix(dataset1)
dataset2_matrix, data2_labels = create_io_matrix(dataset2)
dataset3_matrix, data3_labels = create_io_matrix(dataset3)
############# initial data visualization
'''
df_1 = DataFrame(dataset1_matrix, columns=['x1', 'x2'])
df_1.plot(kind='scatter', x='x1', y='x2')

print("=================== data 1 ======================")
plt.show()

df_2 = DataFrame(dataset2_matrix, columns=['x1', 'x2'])
df_2.plot(kind='scatter', x='x1', y='x2')

print("===================== data 2 ===================")
plt.show()

df_3 = DataFrame(dataset3_matrix, columns=['x1', 'x2'])
df_3.plot(kind='scatter', x='x1', y='x2')

print("================= data 3 ==================")
plt.show()
'''

def K_means_helper(K, input_matrix, initial_means=False):
    N, d = np.shape(input_matrix)
    if isinstance(initial_means, bool):
        """
        *** fill in this blank ***
        randomly initialize the cluster means, if initial_means = False,
        which means you don't pass the initial values manually.
        """
        cluster_means = []
        for k in range(K):
            cluster_means.append(np.random.rand(1, 2))
    else:
        cluster_means = initial_means
    ## get cluster assignments
    cluster_assignments = get_cluster_assignments(K, cluster_means, input_matrix)
    # compute sse
    sse = compute_sse(input_matrix, cluster_assignments, cluster_means, K)
    converged = False
    i = 0
    while not converged:
        prev_cluster_assignments = cluster_assignments
        prev_means = cluster_means
        prev_sse = sse
        # compute the new cluster means
        cluster_means = get_cluster_means(K, cluster_assignments, input_matrix)
        # iterate for one more step
        cluster_assignments = get_cluster_assignments(K, cluster_means, input_matrix)
        sse = compute_sse(input_matrix, cluster_assignments, cluster_means, K)
        ## check if it converges already
        converged = converged_check(prev_cluster_assignments, cluster_assignments, cluster_means, prev_means, prev_sse,
                                    sse, i)
        i = i + 1

    return cluster_assignments, cluster_means, sse


## this function assigns a class label to each data point
def get_cluster_assignments(K, cluster_means, input_matrix):
    """
    K is the number of clusters; cluster_means contains the mean for each cluster; input_matrix is just the input.
    """
    N, d = np.shape(input_matrix)

    cluster_assignments = np.zeros((N, K))
    ### the rows are one hot vectors which indicate whether the data point is in the cluster

    for i in range(N):
        x_i = input_matrix[i]
        """
        *** fill in this blank ***
        you need to compute the Euclidean distance between each data point and all cluster means, and
        assign a class label to each data point. 
        """
        j = 0
        min_dis = float('inf')
        while j < len(cluster_means):
            dis = distance.euclidean(x_i, cluster_means[j])

            if dis < min_dis:
                min_dis = dis
                c_index = j
            j += 1
        cluster_assignments[i][c_index] = 1
    return cluster_assignments


def get_cluster_means(K, cluster_assignments, input_matrix):
    """
    K is the number of clusters; cluster_assignments contains class label for each datapoint;
    input_matrix is just the input.
    """
    X = np.asarray(input_matrix)
    z = np.asarray(cluster_assignments)
    N, d = np.shape(X)
    """
    *** fill in the blank ***
    Based on the cluster assignments, you need recompute the cluster means 
    which could be a K-by-D array/matrix
    and return it
    """
    new_means = []
    for k in range(K):
        count = 0
        temp = np.array([0.0, 0.0])
        for point in getClusteredData(input_matrix, cluster_assignments, k):
            count += 1
            temp += point
        np.true_divide(temp, count)
        new_means.append(temp)
    return new_means


def compute_sse(input_matrix, clustering_assignments, cluster_means, K):
    """
    get the SSE
    """
    SSE_list = []
    for k in range(K):
        cluster_k = getClusteredData(input_matrix, clustering_assignments, k)
        mean = cluster_means[k]
        if len(cluster_k) == 0:
            """
            ** fill in the blank **
            when there is no point in cluster_k
            """
            SSE_list.append(0)
        else:
            """
            ** fill in the blank **
            compute the sse_k for cluster k
            """
            sse_k = 0
            for point in cluster_k:
                sse_k += distance.euclidean(point, mean) ** 2
            SSE_list.append(sse_k)
    return sum(SSE_list)


def getClusteredData(input_matrix, cluster_assignments, k):
    """
    gets all points in cluster k
    """
    wanted = cluster_assignments[:, k]
    wanted_X = []
    for i in range(len(wanted)):
        if cluster_assignments[i][k] == 1:
            wanted_X.append(input_matrix[i])
    return wanted_X

def converged_check(cluster_assignments_prev, cluster_assignments, cluster_means, cluster_means_prev, prev_sse, sse, i):
    """
    *** fill in this blank ***
    Returns a boolean that indicates if the algorithm converges.
    """
    return cluster_assignments_prev.all() == cluster_assignments.all()


def k_means(K, input_matrix, initial_means = False, num_restarts = 300):
    """
    implements the k-means clustering algorithm with multiple iterations
    """

    #for i in range(num_restarts):
    """
    ** fill in the blank **
    run k_means and store the necessary infos after convergence
    and Return the the result with minimal SSE
    """
    best_cluster_assignments, best_cluster_means, best_sse = K_means_helper(K, input_matrix)
    return best_cluster_assignments, best_cluster_means, best_sse



k_means(3, dataset1_matrix)