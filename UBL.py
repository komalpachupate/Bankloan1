#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import os


# In[2]:


os.chdir(r'C:\Users\Mummy\Desktop\KP')


# In[3]:


data=pd.read_csv('UniversalBank.csv')


# In[4]:


data.drop(['ID','ZIP Code'],axis=1,inplace=True)


# In[5]:


X=data.drop(['Personal Loan'],axis=1)
y=data['Personal Loan']


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

filename = 'ubl.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:




