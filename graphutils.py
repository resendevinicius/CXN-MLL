# Author @Vinícius Resende
# Contact: resendeviniciush@gmail.com

import numpy as np
from bisect import bisect_right
from sklearn.metrics.pairwise import euclidean_distances

def knn_graph(X, y, k):
    """Implementation of kNN Graph for multi-label learning

    Args:
        X (scipy.sparse): input data
        y (scipy.sparse): input data set of labels
        k (int): max number of neighbors for each sample

    Returns:
        list: adjacency list of each one of the |L| graphs
        |L| = y.shape[1]
    """
    if isinstance(y, np.ndarray) == False:
        y = y.toarray()

    distances = euclidean_distances(X, X)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis = 1)[:,:k]
    graphs = []

    for label in range(y.shape[1]):
        adj_list = []
        for u in range(X.shape[0]):
            adj_list.append([[u, u]])
            if y[u][label] == 1:
                neighborhood = (neighbors[u][np.transpose(y[neighbors[u]])[label] == 1])
                adj_list.append([[u, v] for v in neighborhood])

        
        adj_list = [item for sublist in adj_list for item in sublist]

        graphs.append(adj_list)

    return graphs

def sknn_graph(X, y, k):
    """Implementation of skNN Graph for multi-label learning

    Args:
        X (scipy.sparse): input data
        y (scipy.sparse): input data set of labels
        k (int): max number of neighbors for each sample

    Returns:
        list: adjacency list of each one of the |L| graphs
    """
    if isinstance(y, np.ndarray) == False:
        y = y.toarray()

    neighbors = [None] * y.shape[1]
    instances = [None] * y.shape[1]

    for label in range(y.shape[1]):
        instances[label] = np.where(y[:, label] == 1)[0]
        neighbors[label] = instances[label][np.argsort(euclidean_distances(X[instances[label]], 
            X[instances[label]]), axis = 1)[:, 1:k + 1]]


    edge_list = [[i, i] for i in range(X.shape[0])]
    graphs = [edge_list.copy() for i in range(y.shape[1])]

    for label in range(y.shape[1]):
        for u, instance in enumerate(instances[label]):
            for v in neighbors[label][u]:
                graphs[label].append([instance, v])

    return graphs

def en_graph(X, y, e):
    """Implementation of eN Graph for multi-label learning

    Args:
        X (scipy.sparse): input data
        y (scipy.sparse): input data set of labels
        e (float): max distance between neighbors

    Returns:
        list: adjacency list of each one of the |L| graphs
    """
    if isinstance(y, np.ndarray) == False:
        y = y.toarray()

    distances = euclidean_distances(X, X)
    e *= np.mean(distances)
    np.fill_diagonal(distances, np.inf)
    neighbors = []
    
    for instance in range(X.shape[0]):
        ordered_distance = np.argsort(distances[instance])
        radius = bisect_right(distances[instance][ordered_distance], e)
        neighbors.append(ordered_distance[:radius])
    
    graphs = []


    for label in range(y.shape[1]):
        adj_list = []
        for u in range(X.shape[0]):
            adj_list.append([[u, u]])
            if y[u, label] == 1:
                neighborhood = (neighbors[u][np.transpose(y[neighbors[u]])[label] == 1])
                adj_list.append([[u, v] for v in neighborhood])

            
        adj_list = [item for sublist in adj_list for item in sublist]

        graphs.append(adj_list)

    return graphs

def knn_en_graph(X, y, k, e):
    """Implementation of kNN + eN Graph for multi-label learning

    Args:
        X (scipy.sparse): input data
        y (scipy.sparse): input data set of labels
        k (int): max number of neighbors for each sample
        e (float): max distance between neighbors
    Returns:
        list: adjacency list of each one the |L| graphs
    """    
    a = knn_graph(X, y, k)
    b = en_graph(X, y, e)

    return [a[label] + b[label] for label in range(y.shape[1])]

def sknn_en_graph(X, y, k, e):
    """Implementation of skNN + eN Graph for multi-label learning

    Args:
        X (scipy.sparse): input data
        y (scipy.sparse): input data set of labels
        k (int): max number of neighbors for each sample
        e (float): max distance between neighbors
    Returns:
        list: adjacency list of each one of the |L| graphs
    """    
    a = sknn_graph(X, y, k)
    b = en_graph(X, y, e)

    return [a[label] + b[label] for label in range(y.shape[1])]