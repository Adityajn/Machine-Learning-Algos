#Preprocessing tut
import pandas as pd 
import numpy as np 
from sklearn import preprocessing as prep

df = pd.read_csv('mydata.csv')

# label encoder to transform categorical string data to integers
df.Sex.unique() #array(['male', 'female'], dtype=object)

le = prep.LabelEncoder()
le.fit(concat.Sex)

Sex_le = le.transform(df.Sex)
df.Sex = Sex_le

