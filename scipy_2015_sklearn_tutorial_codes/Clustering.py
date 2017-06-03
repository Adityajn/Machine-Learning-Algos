import matplotlib.pyplot as plt
import numpy as np

"""
Clustering is the gathering samples into group of similar samples according to some
predefined similarity and dissimilarity measure

"""

from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42)
plt.scatter(X[:, 0], X[:, 1])

"""
KMeans is one of very basic clustering algorithms


"""

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)


"""
The obtained labels may not be as same as y. but may have different name for different clusters
Therefore accuracy_score may be 0
Confusion_matrix show how classes have been alloted to clusters
        [[ 0  0 34]
        [33  0  0]
        [ 0 33  0]]
        In this example class 0 has label as 3
        class 1 has label 0
        class 2 has label 1
"""
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(y, labels))
print(confusion_matrix(y, labels))

"""
adjusted_rand_score will give how well our clusters are.

"""
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y, labels)



"""
There are different clustering algorithms each has its drawback and advantages

sklearn.cluster.KMeans
    This works only when cluster are grouped together which has std same in every direction.
    It cannot detect one side longeted clusters
sklearn.cluster.MeanShift
    Better than KMeans but doesnt work with large samples
sklearn.cluster.DBSCAN:
    Can detect irregularly shaped clusters based on density,
    i.e. sparse regions in the input space are likely to become inter-cluster boundaries.
    Can also detect outliers (samples that are not part of a cluster).
sklearn.cluster.AffinityPropagation:
    Clustering algorithm based on message passing between data points.
sklearn.cluster.SpectralClustering:
    KMeans applied to a projection of the normalized graph Laplacian:
    finds normalized graph cuts if the affinity matrix is interpreted as an adjacency matrix of a graph.
sklearn.cluster.Ward:
    Ward implements hierarchical clustering based on the Ward algorithm, a variance-minimizing approach.
    At each step, it minimizes the sum of squared differences within all clusters (inertia criterion).

Of these, Ward, SpectralClustering, DBSCAN and Affinity propagation can also work with precomputed similarity matrices.
"""
