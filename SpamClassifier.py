# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:09:43 2020

@author: NISHIT JOSHI
"""


import pandas as pd

data = pd.read_csv("F:\Data\SMSSpamCollection", sep='\t',names=["Label","message"])


#data cleaning

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


ps = PorterStemmer()
corpus = []

for i in range(0,len(data)):
    review = re.sub('[^a-zA-Z]', ' ' ,data['message'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
x = cv.fit_transform(corpus).toarray()

#dependent var
y = pd.get_dummies(data['Label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB().fit(x_train,y_train)

y_pred = nb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
a = accuracy_score(y_test,y_pred)





