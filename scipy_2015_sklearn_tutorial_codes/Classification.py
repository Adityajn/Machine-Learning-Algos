import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

X,y=make_blobs(n_samples=200,n_features=2,centers=2,cluster_std=3,center_box=(-10,10))
# generate a isotonic guassian blob for clustering
print(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1999)

plt.scatter(X_train[:,0], X_train[:, 1], c=y_train)
plt.title("Training Data")
plt.xlabel('feature_1')
plt.ylabel('feature_2')

"""
#LogisticRegression and Prediction
#logistic regression is called linear model it will separate two inputs with a straight line
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
"""

#KNeginbour classifiers
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=10);
classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)


import plot_2d_separator as pd      #special function
pd.plot_2d_separator(classifier,X_train)

#Accuracy
print(" accuracy = {} ".format(np.sum(y_predict==y_test)/np.size(y_test)))
# or np.mean(y_predict==y_test)
# or classifier.score(X_test,y_test)

plt.figure()
#visualize test results

plt.scatter(X_test[y_test==y_predict][:,0],X_test[y_predict==y_test][:,1],c=y_test[y_predict==y_test],marker='o')
plt.scatter(X_test[y_test!=y_predict][:,0],X_test[y_predict!=y_test][:,1],c='black',marker='x')
plt.title("Testing Data")
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()

print(classifier.classes_)