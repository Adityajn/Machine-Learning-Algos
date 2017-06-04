"""
X = ["Some say the world will end in fire,",
     "Some say in ice."]
        |
        |   Tokenizer
        |
X= [['some','say','the','world','will','end','in','fire'],['some','say','in','ice']]
        |
        |   Build Dictionary
        |
['some','say','the','world','will','end','in','fire','ice']
        |
        |   Sparse Matrix
        |
X_BAG_OF_WORDS = [[1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1]]
"""



X = ["Some say the world will end in fire,",
     "Some say in ice."]


# CountVectorizer is most common algorithm
from sklearn.feature_extraction.text import CountVectorizer
vectorizer= CountVectorizer()
vectorizer.fit(X)

#print(vectorizer.vocabulary_)

X_BAG_OF_WORDS = vectorizer.transform(X)

#print(X_BAG_OF_WORDS.toarray())
#print(vectorizer.inverse_transform(X_BAG_OF_WORDS))

print(vectorizer.get_feature_names())
#['end', 'fire', 'ice', 'in', 'say', 'some', 'the', 'will', 'world']



"""

Tfidf algorithm - term-frequency inverse-document-frequency (Tfidf) scaling
Normal text is treated as shorter text
example 'the' appears in most of documents quite often, so it will have less weight

"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(X)

import numpy as np
np.set_printoptions(precision=2)
print(tfidf.transform(X).toarray())
"""
[[ 0.39  0.39  0.    0.28  0.28  0.28  0.39  0.39  0.39]
 [ 0.    0.    0.63  0.45  0.45  0.45  0.    0.    0.  ]]
"""

"""
Bigrams and N-grams is used when not as single token
but sequence of tokens are used

if 2 token are used it is bigram otherwise n-grams
"""

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2,2)) #min=2 max=2
vectorizer.fit(X)
print(vectorizer.get_feature_names())


