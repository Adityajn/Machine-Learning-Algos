import os
import numpy as np
import matplotlib.pyplot as plt


data = list()
target = list()
with open(os.path.join("datasets","smsspam","SMSSpamCollection")) as f:
    for line in f:
        type,sms=line.strip().split("\t")
        data.append(sms)
        target.append(type=='ham')



#split data and target
from sklearn.cross_validation import train_test_split
data_train,data_test,y_train,y_test  = train_test_split(data,target,train_size=.75,random_state=42)


#Lets get Bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(data_train)

X_train=vectorizer.transform(data_train)
X_test=vectorizer.transform(data_test)

len(vectorizer.vocabulary_)     #7532
X_train.shape       #(4180, 7532)
X_test.shape        #(1394, 7532)


#train a classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)

print("Accuracy CV : {}".format(classifier.score(X_test,y_test)))


"""
With countVectorizer and document freq equal 2
"""
vectorizer2=CountVectorizer(min_df=2)       #minimum doc freq means word must be in 2 doc to be included in dict
vectorizer2.fit(data_train)
X_train=vectorizer2.transform(data_train)
X_test=vectorizer2.transform(data_test)

classifier2=LogisticRegression()
classifier2.fit(X_train,y_train)
print("Accuracy CV and min_df=2: {}".format(classifier2.score(X_test,y_test)))

def visualize_coefficients(classifier, feature_names, n_top_features=25):
    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
    plt.bar(np.arange(50), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 51), feature_names[interesting_coefficients], rotation=60, ha="right")
    plt.show()
visualize_coefficients(classifier2,vectorizer2.get_feature_names())


"""
With Tfidf Vectorizer
"""
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(data_train)
X_train=tfidf.transform(data_train)
X_test=tfidf.transform(data_test)

classifier2=LogisticRegression()
classifier2.fit(X_train,y_train)
print("Accuracy Tfidf : {}".format(classifier2.score(X_test,y_test)))


"""
With Tfidf Vectorizer
"""
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=2)
tfidf.fit(data_train)
X_train=tfidf.transform(data_train)
X_test=tfidf.transform(data_test)

classifier2=LogisticRegression()
classifier2.fit(X_train,y_train)
print("Accuracy  tfidf and min_df=2: {}".format(classifier2.score(X_test,y_test)))


"""
With count frequency and n
"""
vectorizer2=CountVectorizer(ngram_range=(1,2))       #minimum doc freq means word must be in 2 doc to be included in dict
vectorizer2.fit(data_train)
X_train=vectorizer2.transform(data_train)
X_test=vectorizer2.transform(data_test)

classifier2=LogisticRegression()
classifier2.fit(X_train,y_train)
print("Accuracy CV and ngram_range=(1,2) : {}".format(classifier2.score(X_test,y_test)))
