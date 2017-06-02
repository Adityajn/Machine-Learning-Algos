import matplotlib.pyplot as plt
import numpy as np

#lets create a training set with only one feature
X = np.linspace(-3,3,100) #features
rng = np.random.RandomState(42)
y = np.sin(X*4)+X+rng.uniform(size=len(X))    #labels

#lets visualize feature vs label (only possible because there is only one feature)
#visualization only possible in 1 and 2 features
plt.scatter(X, y, c='b', marker='o')

#features must be 2d arrray
print(X.shape)
X=X[:, np.newaxis]
print(X.shape)

#divide test and training data
from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=.75, random_state=42)

"""
#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X,train_y)
predict_y = regressor.predict(test_X)
"""

#KNeighbour Regression
from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor(n_neighbors=3)
regressor.fit(train_X,train_y)
predict_y=regressor.predict(test_X)

#visualize test result
plt.figure()
plt.scatter(test_X, test_y, c='b', marker='o',label='data')
plt.scatter(test_X, predict_y, c='g', marker='^',label='prediction')
plt.legend(loc='best')
#plt.show()

#Accuracy
print(regressor.score(test_X,test_y))










