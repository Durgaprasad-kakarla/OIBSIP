# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 20:40:26 2022

@author: wwwka
"""

import pandas as pd

df=spam_detectioncsv.iloc[:,:2]

# Data Cleaning and preprocessing
import re
import nltk
 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(len(df['v2'])):
    review=re.sub('[^a-zA-Z]',' ',df['v2'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
# Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(df['v1'])
y=y.iloc[:,:1]

#Train the model

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
y_pred=rf.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(accuracy_score(y_pred,y_test))