
# coding: utf-8

# # DS5230/DS4220--Unsupervised Machine Learning and Data Mining – Fall 2018 – Homework 3 -- Implement Clustering Algorithms

# In this exercise, you are required to implement K-means and DBSCAN algorithms from scratch. I wrote this draft which gives you some instructions on how to code it up step by step.
# Notice that you are NOT required to use this framework, please free feel to come up with your own implementation.

# ### Load Dataset 1, 2, 3

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.spatial import *


# In[2]:


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


# In[3]:


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

df_1 = DataFrame(dataset1_matrix, columns = ['x1', 'x2'])
df_1.plot(kind='scatter', x = 'x1', y = 'x2')

print("=================== data 1 ======================")
plt.show()

df_2 = DataFrame(dataset2_matrix, columns = ['x1', 'x2'])
df_2.plot(kind='scatter', x = 'x1', y = 'x2')

print("===================== data 2 ===================")
plt.show()

df_3 = DataFrame(dataset3_matrix, columns = ['x1', 'x2'])
df_3.plot(kind='scatter', x = 'x1', y = 'x2')

print("================= data 3 ==================")
plt.show()


# 
# 

# ## DBSCAN Algorithm (Q4)
# Firstly let's implement DBSCAN algorithm. The following two functions could potentially be helpful for the implementation.
# member_cluster(q, clusters) take a data point (one row in dataset_matrix) and dictionary type variable called clusters, returns a boolean indicating whether this point has already been assigned to a cluster. You might wonder why I convert q to list(q), that's just a technical trick, since the values in 'clusters' are lists of lists,i.e. each point was converted to list type and then put into the dictionary

# In[4]:


def member_cluster(q, clusters):
    """
    checks if the point (q) has already been assigned a cluster 
    """
    for _, c in clusters.items(): 
        if list(q) in c:
            return True
    return False

def has_cluster(q, clusters):
    return tuple(q) in clusters.keys()
    
def region_query(p, eps, input_matrix): 
    """
    :return: all points within P's eps-neighborhood (including P)
    """
    eps_neighbors = [list(x) for x in input_matrix if (distance.euclidean(x, p) <= eps)]
    return eps_neighbors


# In[33]:


def update_label(point, noise, visited, clusters, c_index):
    clusters[tuple(point)] = c_index
    if point in visited:
        visited.remove(point)
    if point in noise:
        noise.remove(point)


# Next you need to implement the algorithm. The first function can put the points in a cluster, and return the corresponding arguments it takes.

# In[46]:


def expand_cluster(p, neighbors, eps, min_points, visited, clusters, c_index, input_matrix, noise): 
    """
    ** fill in the blank **
    gets the points in a cluster  
    """ 
    i = 0
    clusters[tuple(p)] = c_index
    while i < len(neighbors):
        point = neighbors[i]
        if point in noise:
            clusters[tuple(point)] = c_index
            visited.append(point)
            noise.remove(point)
        elif (point in visited) & ~has_cluster(point, clusters):
            clusters[tuple(point)] = c_index
            neighbors_new = region_query(point, eps, input_matrix)
            if len(neighbors_new) >= min_points:
                neighbors += neighbors_new
        i += 1
    return visited, clusters, noise


# Finally you can implement algorithm in a way that it goes through all points in the dataset and performs what it is supposed to do

# In[49]:


def DBSCAN(eps, min_points, input_matrix): 
    """
    implements the DBSCAN clustering algorithm 
    """
    visited = []
    noise = []
    clusters = {}
    c_index = 0
    
    print("running DBSCAN with eps = %f, and min_pts = %f" % (eps, min_points))
  
    for p in input_matrix:
        if list(p) in visited: 
            """
            ** fill in the blank **
            """
            continue
        if list(p) not in visited:
            visited.append(list(p))
            neighbors = region_query(p, eps, input_matrix) 
            if len(neighbors) < min_points:
                """
                ** fill in the blank **
                """
                noise.append(list(p))
                #visited.remove(list(p))
            else: 
                """
                ** fill in the blank **
                ## Hint : need to call expand_cluster here
                """ 
                '''
                if has_cluster(p, clusters):
                    visited, clusters, noise = expand_cluster(p, neighbors, eps, min_points, visited, clusters, clusters[tuple(p)], input_matrix, noise)
                else:
                    visited, clusters, noise = expand_cluster(p, neighbors, eps, min_points, visited, clusters, c_index + 1, input_matrix, noise)
                '''
                c_index += 1
                visited, clusters, noise = expand_cluster(p, neighbors, eps, min_points, visited, clusters, c_index, input_matrix, noise)
    return visited, clusters, noise


# ### 1) Run DBSCAN with different eps and min_pts
# ### 2) Evaluation of DBSCAN with different metrics for each eps and min_pts. Report the results.

# In[50]:


print("==============DBSCAN for dataset 1==============")
# run experiments
visited, clusters, noise = DBSCAN(0.2, 2, dataset1_matrix)
clusters
## evaluations


# In[18]:


list(clusters.keys())[0]


# In[13]:


plt.scatter(x=clusters.keys(), color=list(clusters.values()))


# In[50]:


from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.2, min_samples=2, metric='euclidean').fit(dataset1_matrix)
clustering.labels_


# In[ ]:


print("==============DBSCAN for dataset 1==============")
# run experiments
## evaluations


# In[ ]:


print("==============DBSCAN for dataset 1==============")
# run experiments
## evaluations


# ### For each dataset, visualize the best result (based on NMI) make a scatter plot.

# In[ ]:


## visualization


# ### K-means implementation (Q5): 
# Now we implement K-means algorithms. K_means_helper is supposed to run k_means given the input with random initialization. 

# In[18]:


def K_means_helper(K, input_matrix, initial_means = False): 
    N, d = np.shape(input_matrix)
    if isinstance(initial_means, bool): 
        """
        *** fill in this blank ***
        randomly initialize the cluster means, if initial_means = False,
        which means you don't pass the initial values manually.
        """
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
        converged = converged_check(prev_cluster_assignments, cluster_assignments, cluster_means, prev_means, prev_sse, sse, i)
        i = i + 1
        
    return cluster_assignments, cluster_means, sse


# Define a function that assigns class label to each data point based on the current cluster means.

# In[ ]:


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
      
    return cluster_assignments


# Then you need a function that computes new cluster means.

# In[14]:


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
    return new_means


# Next you need to compute the overall SSE. finish the definition of function compute_see.
# To make your life easier, the function 'getClusteredData' can help you grab all points in cluster k 
# as long as you pass the right arguments.

# In[ ]:


def compute_sse(input_matrix, clustering_assignments, cluster_means, K): 
    """
    get the SSE 
    """
    SSE_list = []
    for k in range(K): 
        cluster_k =  getClusteredData(input_matrix, clustering_assignments, k)
        if len(cluster_k) == 0: 
             """
             ** fill in the blank **
             when there is no point in cluster_k
             """
        else: 
             """
             ** fill in the blank **
             compute the sse_k for cluster k
             """
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


# In[15]:


def converged_check(cluster_assignments_prev, cluster_assignments, cluster_means, cluster_means_prev, prev_sse, sse, i): 
    """
    *** fill in this blank ***
    Returns a boolean that indicates if the algorithm converges.  
    """
    return 


# Finally should preform multiple random initializations and keep the best 

# In[21]:


def k_means(K, input_matrix, initial_means = False, num_restarts = 300):
    """
    implements the k-means clustering algorithm with multiple iterations
    """

    for i in range(num_restarts): 
        """
        ** fill in the blank **
        run k_means and store the necessary infos after convergence
        and Return the the result with minimal SSE
        """
    return best_cluster_assignments, best_cluster_means, best_sse


# ### Now you need to visualize optimal result in a way that points in different clusters are colorcoded differently.

# In[16]:


## plotting functions


# ### Execution of k-means with different Ks = 1,2,3,4,5.
# ### For each K, also evaluate the results with all those 3 metrics.

# In[33]:


print("===========dataset 1===========")
#for i in range(4):
    
    ## executions
    ## visualization
    ## evaluation functions
print("===========dataset 2===========")  
#for i in range(4):
    ## executions
    ## visualization
    ## evaluation functions
print("===========dataset 3===========")
#for i in range(4):
    ## executions
    ## visualization
    ## evaluation functions


# ## Question 7: 
# You need to fill in the blanks to answer question 7.

# 
# ### Dataset 1: 
# 
# K-means K=3
# 
#   NMI:
#   
#   SC: 
#   
#   CH: 
#   
# DBScan.
# 
#    NMI:
#    
#    SC: 
#    
#    CH:
# 
# Which performs best for dataset 1? and why?
# Answer :  
# 
# ----------------------------------------------------------------------------------------  
# ### Dataset 2: 
# 
# K-means K=3.
# 
#   NMI: 
#   
#   SC: 
#   
#   CH: 
#   
# DBScan.
# 
#    NMI:
#    
#    SC: 
#    
#    CH:
# 
# Which performs best for dataset 2? and why?
# Answer : 
# 
# ----------------------------------------------------------------------------------------
# ### Dataset 3: 
# 
# K-means K=3.
# 
#   NMI: 
#   
#   SC: 
#   
#   CH: 
#   
# DBScan.
# 
#    NMI:
#    
#    SC: 
#    
#    CH:
#   
# Which performs best for dataset 3? and why?
# Answer : 
