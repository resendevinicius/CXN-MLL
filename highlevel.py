# Copyright @Vinícius Resende
# Contact: resendeviniciush@gmail.com

import numpy as np
import igraph as ig
import scipy
import graphutils
import sklearn.metrics as metrics
from skmultilearn.base import MLClassifierBase
from sklearn.metrics.pairwise import euclidean_distances
from skmultilearn.adapt import MLkNN


class HighLevel(MLClassifierBase):
    
    """HighLevel hybrid classifier for multi-label classification 

    Args:
        k (int): number of vertices each sample will connect
        e (float): e-radius, max distance between neighbors
        weight_l (float): weight to priorize high or low level features
        weight_d (float list): weight for each metric, size has to be 3 (one for each metric) 
            and sum 1, ex:[0.4, 0.4, 0.2]
        threshold (float): minimum probability needed to make a label assignment
        classifier (object): low level classifier that will be used in the combination
        graph_construction_method (int): [1 = knn, 2 = sknn, 3 = en, 4 = knn+en, 5 = sknn+en]

    References:
        If you use this classifier please cite the original paper introducing the method:
        .. code :: bibtex
            @inproceedings{resende2020high,
                title={High-Level Classification for Multi-Label Learning},
                author={Resende, Vin{\'\i}cius H and Carneiro, Murillo G},
                booktitle={2020 International Joint Conference on Neural Networks (IJCNN)},
                pages={1--8},
                year={2020},
                organization={IEEE}
            }
    """

    def __init__(self, k=3, e=0, weight_l=0.3, weight_d=[0.33, 0.33, 0.33], threshold=0.7,
                 classifier=MLkNN(k=5), graph_construction_method=1):

        super(HighLevel, self).__init__()

        self.k = k
        self.e = e
        self.weight_l = weight_l
        self.weight_d = weight_d
        self.threshold = threshold
        self.classifier = classifier
        self.graph_construction_method = graph_construction_method
        self.copyable_attrs += ["k", "e", "weight_l",
                                "weight_d", "threshold", "classifier"]

    def build_graph(self, X, y):
        """Build the graph from the training set (default = supervised knn-graph)

        Args:
            X (scipy.sparse): training set instances
            y (scipy.sparse): set of labels of each training set instance
        """
        adj_list = []
        self.graphs = []
        if self.graph_construction_method == 1:
            adj_list = graphutils.knn_graph(X, y, self.k)
        elif self.graph_construction_method == 2:
            adj_list = graphutils.sknn_graph(X, y, self.k)
        elif self.graph_construction_method == 3:
            adj_list = graphutils.en_graph(X, y, self.e)
        elif self.graph_construction_method == 4:
            adj_list = graphutils.knn_en_graph(X, y, self.k, self.e)
        elif self.graph_construction_method == 5:
            adj_list = graphutils.sknn_en_graph(X, y, self.k, self.e)
        else:
            print('Not a valid graph construction method. \
                {} out of expected range [1, 5]'.format(self.graph_construction_method))

        for label in range(y.shape[1]):
            self.graphs.append(
                ig.Graph(adj_list[label], directed=False).simplify())

    def fit(self, X, y):
        """Fit the model

        Args:
            X (scipy.sparse): training dataset instances
            y (scipy.sparse): set of labels of each sample

        Returns:
            self: fitted instance of the classifier
        """
        
        self.X = X
        self.y = y
        if isinstance(y, np.ndarray) == False:
            self.y = y.toarray()

        self.n_labels = self.y.shape[1]
        self.build_graph(X, y)
        self.metrics = np.zeros((3, self.n_labels), dtype='float32')
        self.proportion = np.zeros(self.n_labels, dtype='float32')
        

        for label in range(self.n_labels):
            self.proportion[label] = len(np.where(self.y[:, label] == 1)[0]) / X.shape[0]
            self.metrics[0][label] = ig.Graph.assortativity_degree(self.graphs[label])
            self.metrics[1][label] = ig.Graph.transitivity_avglocal_undirected(self.graphs[label])
            self.metrics[2][label] = ig.mean(self.graphs[label].degree())

        self.metrics = np.nan_to_num(self.metrics)
        return self

    def compute_high_level_proba(self, neighbors):
        """Compute the probability of label assignments for 1 instance

        Args:
            neighbors (array): training set neighbors of the test item

        Returns:
            array: probability of the test item be classified in each class
        """
        c = np.where(self.y[neighbors] == 1)
        labels = np.unique(c[1])

        [self.graphs[label].add_vertices(1) for label in labels]
        connect = [[self.X.shape[0], neighbors[c[0][i]], c[1][i]] for i in range(c[0].shape[0])]
        [self.graphs[label].add_edge(u, v) for u, v, label in connect]

        remove = np.arange(self.n_labels)[[value not in labels for value in range(self.n_labels)]]

        metrics = np.zeros((3, self.n_labels), dtype='float32')
        min_degree = np.zeros(self.n_labels)
        max_degree = np.zeros(self.n_labels)
        for label in range(self.n_labels):
            metrics[0][label] = ig.Graph.assortativity_degree(self.graphs[label])
            metrics[1][label] = ig.Graph.transitivity_avglocal_undirected(self.graphs[label])
            degree = self.graphs[label].degree()
            min_degree[label] = np.min(degree)
            max_degree[label] = np.max(degree)
            metrics[2][label] = ig.mean(degree)

        metrics = np.nan_to_num(metrics, nan = 0)
        variation = np.absolute(metrics - self.metrics)
        variation[0][remove] = 2
        variation[1][remove] = 1
        variation[2][remove] = [max(metrics[2][label] - min_degree[label], max_degree[label]
                                    - metrics[2][label]) for label in remove]

        
        sum_normalize = lambda x : x / sum(x)
        metrics = np.arange(3)
        proba = np.zeros(self.n_labels, dtype='float32')
             
        for metric in metrics:
            variation[metric] = sum_normalize(variation[metric])
       
        for label in range(self.n_labels):
            if label in labels:
                self.graphs[label].delete_vertices(self.X.shape[0])

        for metric in metrics:
            for label in labels:
                variation[metric][label] *= self.proportion[label]

        for metric, weight in enumerate(self.weight_d):
            variation[metric] = weight * np.array(1.0 - variation[metric])

        proba = np.array(variation[0] + variation[1] + variation[2])

        return proba


    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Args:
            X (scipy.sparse): representation of the training set instances

        Returns:
            scipy.sparse: the probabilities
        """
        neighbors = np.argsort(euclidean_distances(X, self.X))[:, :self.k]
        results = scipy.sparse.lil_matrix((X.shape[0], self.n_labels), dtype='float')
        self.tc_proba = self.classifier.fit(self.X, self.y).predict_proba(X)


        for instance in range(X.shape[0]):
            results[instance] = self.compute_high_level_proba(neighbors[instance]) \
                * self.weight_l + self.tc_proba[instance] * (1 - self.weight_l)

        return results

    def predict(self, X):
        """Predict the label assignments for X

        Args:
            X (scipy.sparse): representation of the training set instances

        Returns:
            scipy.sparse: binary indicator matrix with label assignments
        """

        return self.predict_proba(X) >= self.threshold

 
