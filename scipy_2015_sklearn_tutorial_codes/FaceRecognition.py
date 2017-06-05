from sklearn import datasets
face_data = datasets.fetch_lfw_people(min_faces_per_person=70,resize=0.4,data_home='datasets')
face_data.keys()     #dict_keys(['DESCR', 'target', 'data', 'target_names', 'images'])

#Lets see waht we have
print("{} , {} , {}".format(face_data.data.shape, face_data.target.shape, face_data.images.shape))
# face_data.data.shape - (1288, 1850)
# face_data.target.shape (1288,)
# face_data.images.shape (1288, 50, 37)

#Lets get data
X = face_data.data
y = face_data.target
names = face_data.target_names

#Lets see photo
import matplotlib.pyplot as plt
plot = plt.figure(figsize=(5,2))
plot.subplots_adjust(left=0,right=1,top=1,bottom=0,hspace=0.05,wspace=0.05)
for i in range(10):
    ax=plot.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(X[i].reshape(50,37),cmap=plt.cm.bone,interpolation='nearest')
    #ax.text(0,38,names[y[i]],color='b')


#lets seperate train and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1999)

#We will use SVM for which 1850 are too many features, lets reduce features and feed into SVM
#we will use Randomiced PCA to reduce features

from sklearn import decomposition
pca = decomposition.RandomizedPCA(n_components=150,whiten=True,random_state=1999)
pca.fit(X_train)
X_train_trans = pca.transform(X_train)
X_test_trans = pca.transform(X_test)
decomposition.RandomizedPCA()

"""
1850 features are represented by 150 components
Each component is called eigenface. So we have 150 eigenfaces
A photo of a person is represented using combination of these eigenfaces
Eigenfaces can be obtained by pca.components_
pca.components_.shape is 150,1850
"""

print("{} {} {}".format(pca.mean_.shape,pca.components_.shape,X_train_trans.shape))
#(1850,) (150, 1850) (966, 150)

#Lets see how many faces we have per sample
import numpy as np
unique_faces=np.unique(y_train)
counts = [ np.sum(y_train==i) for i in unique_faces]
print(counts)
plt.figure()
plt.bar(unique_faces,counts)


#lets see mean face
plt.figure(figsize=(2,2))
plt.imshow(pca.mean_.reshape(50,37),cmap=plt.cm.bone)

#Lets see all some eigenfaces
eigenfaces = pca.components_
plot2= plt.figure(figsize=(5,8))
plot2.subplots_adjust(left=0,right=1,top=1,bottom=0,hspace=.05,wspace=.05)
for i in range(15):
    ax = plot2.add_subplot(3,5,i+1)
    ax.imshow(eigenfaces[i].reshape(50,37),cmap=plt.cm.bone,interpolation='nearest')


#Lets train a SVM classifier
from sklearn.svm import SVC
svm=SVC(C=5., gamma=0.001)
svm.fit(X_train_trans,y_train)
y_predict=svm.predict(X_test_trans)

print(svm.score(X_test_trans,y_test))

#plt.show()