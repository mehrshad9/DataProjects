#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing

# ## Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[ ]:


df1 = pd.read_csv('training.1600000.processed.noemoticon.csv' , encoding= 'latin')


# In[ ]:


dataset=df1.sample(frac=0.01, replace=False, random_state=19)
dataset.reset_index(inplace=True)


# In[ ]:


dataset.shape


# In[ ]:


dataset.columns=['index' , 'sentiment' ,'ID' , 'date' , 'NQ' , 'id' , 'text']


# In[ ]:


dataset.columns


# In[ ]:


dataset.drop(['index', 'ID' , 'date' , 'NQ' , 'id'] , axis=1 , inplace=True)


# In[ ]:


dataset.head()


# In[ ]:


dataset.sentiment.value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
dataset['sentiment'] =lb.fit_transform(dataset['sentiment'])


# In[ ]:


dataset.sentiment


# ## Cleaning the texts

# In[ ]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []


# In[ ]:



for i in range(0, len(dataset)):
   review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
   review = review.lower()
   review = review.split()
   ps = PorterStemmer()
   all_stopwords = stopwords.words('english')
   review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
   review = ' '.join(review)
   corpus.append(review)


# In[ ]:


all_stopwords


# In[ ]:


corpus[-1]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)


# ## Creating the Bag of Words model

# In[ ]:



X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values


# In[ ]:


X.shape


# ## Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Training the Naive Bayes model on the Training set

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=40)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# ## Predicting the Test set results

# ## Making the Confusion Matrix

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

