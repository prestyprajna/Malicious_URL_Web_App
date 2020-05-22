# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:37:10 2020

@author: Prajna
"""

#import os
#os.chdir('C:\Users\Prajna\Desktop\new_log_reg')

#import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle

#removing index column and replace '??' and '###' as missing values
#df=pd.read_csv('C:/Users/Prajna/Desktop/dataset.csv',index_col=0,na_values=["??","###"])
df=pd.read_csv('dataset.csv',index_col=0)
df.head()

numerical_data=df.select_dtypes(exclude=[object])
print(numerical_data.shape)

df['CONTENT_LENGTH'] = df['CONTENT_LENGTH'].fillna(0)

from sklearn import preprocessing
# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
df['SERVER'] = le.fit_transform(df['SERVER'])

#df['CONTENT_LENGTH'].astype('int64')

#print(df.info())

#print(np.unique(df['CONTENT_LENGTH']))


#used to obtain correlation between numerical data types
correlated_data = numerical_data.corr()
correlated_data

relevant_features=(correlated_data > 0.90)
relevant_features

import sklearn
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# select needed columns 
#cols = [col for col in df.columns if col  in ['URL_LENGTH','NUMBER_SPECIAL_CHARACTERS','CONTENT_LENGTH','SERVER','APP_BYTES']]   
                                       
#cols = [col for col in df.columns if col  in ['URL_LENGTH','NUMBER_SPECIAL_CHARACTERS']]                                           
cols = [col for col in df.columns if col  in ['URL_LENGTH','NUMBER_SPECIAL_CHARACTERS','CONTENT_LENGTH','SERVER']]   

# dropping the 'Type' column
data = df[cols]

#assigning the Target column as target
target = df['Type']

data.head(n=2)

# create a base classifier used to evaluate a subset of attributes(RFE ALGORITHM)
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 1)
rfe = rfe.fit(data,target)
# summarize the selection of the attributes
#print(rfe.support_)
print(rfe.ranking_)

first  = df.loc[df['Type'] == 0]  #used to select rows whose Target value is 0
second = df.loc[df['Type'] == 1]  #used to select rows whose Target value is 1

import matplotlib.pyplot as plt
#matplotlib inline
plt.xlabel('URL_LENGTH')
plt.ylabel('NUMBER_SPECIAL_CHARACTERS')
plt.scatter(first['URL_LENGTH'], first['NUMBER_SPECIAL_CHARACTERS'],color="green",marker='+')
#plt.scatter(second['URL_LENGTH'], second['NUMBER_SPECIAL_CHARACTERS'],color="blue",marker='.')


#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.metrics import f1_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score
#from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(data, target.ravel(), test_size=0.4,random_state=10)

"""
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=500)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
cm = confusion_matrix(y_test, y_pred) 
print(cm)
print ('Accuracy:', accuracy_score(y_test, y_pred))

from sklearn.metrics import cohen_kappa_score
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred))
#print(y_test.shape)

"""
#RANDOM FOREST 
from sklearn.ensemble import RandomForestClassifier  
model1 = RandomForestClassifier(min_samples_split=10) 
model1.fit(X_train, y_train) 
y_pred = model1.predict(X_test)
cm = confusion_matrix(y_test, y_pred) 
print(cm)
print ('Accuracy:', accuracy_score(y_test, y_pred))

from sklearn.metrics import cohen_kappa_score
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred))


# Saving model to disk
pickle.dump(model1, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
