# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:59:36 2020

@author: Prajna
"""

def calculate_ser_val(server_name):
    
    import pandas as pd
                        
    df=pd.read_csv('dataset.csv',index_col=0)
    #df.head()          
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df['SERVER'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    #print(le_name_mapping)
    server_val = le_name_mapping[server_name]
    return server_val