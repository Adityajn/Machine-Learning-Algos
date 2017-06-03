import  matplotlib.pyplot as plt
import numpy as np

X=input()
"""

One kind of supervised learning is where we want to reperesent our data in some new kind representation
The New Representation will tell more about data than data we had previously

It is used to preprocess your data
ex normalize the data means mean is 0 and std is 1
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


"""
Principal Component Analysis
Used to reduce dimensionality of data
i.e. we find new feaatures using linear combination of old features
"""
from sklearn.decomposition import PCA
from sklearn.decomposition import truncated_svd     #truncatedsvd doesnot remove mean
pca=PCA(n_components=2)     #decompose into 2 components
pca.fit(X)
X_pca=pca.transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1])

"""
Manifold Learning Algorithms
PCA cannot detect non-linear functions. But Manifold Algorithm are capable of this

Isomap and TSNE are 2 powerful algorithm for this.
"""

from sklearn.manifold import Isomap
X_iso = Isomap.fit_transform(X)
plt.scatter(X_iso[:,0],X_iso[: ,1],c=y)

from sklearn.manifold import TSNE
X_tsne = TSNE.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[: , 1], c=y)