from sklearn.datasets import load_boston
boston=load_boston()

boston.keys()
#dict_keys(['DESCR', 'data', 'target', 'feature_names'])

#print(boston.DESCR)
#It is dataset of boston house prices which depend on no of attributes(13)

#print(boston.data.shape)   (506,13)
#print(boston.target.shape)     (506)

from sklearn.cross_validation import train_test_split
train_X,test_X,train_y,test_y = train_test_split(boston.data,boston.target,train_size=0.75,random_state=42)

"""
from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor(n_neighbors=3)
regressor.fit(train_X,train_y)
"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(train_X,train_y)


accuracy=regressor.score(test_X,test_y)
print("Accuracy: {}".format(accuracy))

