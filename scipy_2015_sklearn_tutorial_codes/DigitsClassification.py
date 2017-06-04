import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data        #(1797,64)
y = digits.target      #(1797,)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=.25, random_state=42)


#lets do Kmeans Clustering
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=10)
clusters=kmeans.fit_predict(X_train)

#Lets see Manifold representation
from sklearn.manifold import TSNE
tsne=TSNE()
X_trans = tsne.fit_transform(X_train)

fig,ax=plt.subplots(1,2)
ax[0].scatter(X_trans[:,0],X_trans[:,1],c=y_train)
ax[1].scatter(X_trans[:,0],X_trans[:,1],c=clusters)

"""
#Lets predict test data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)
print("Accuracy : {}".format(classifier.score(X_test,y_test)))


#Lets visualize predicted text
plot = plt.figure(figsize=(6,6))    #starts new figure
plot.subplots_adjust(left=0,right=1,top=1,bottom=0,hspace=0.05,wspace=0.05)
for i in range(64):
    ax = plot.add_subplot(8,8,i+1,xticks=[],yticks=[])
    ax.imshow(X_test[i].reshape(8,8),cmap=plt.cm.binary, interpolation='nearest')
    if y_test[i]==y_predict[i]:
        ax.text(0,7,str(y_predict[i]),color='g')
    else:
        ax.text(0,7,str(y_predict[i]),color='r')
    ax.text(1,0,str(y_test[i]),color='b')



#Lets see where it is doing wrong
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_predict)
print(conf_mat)

plt.matshow(conf_mat)
plt.show()
