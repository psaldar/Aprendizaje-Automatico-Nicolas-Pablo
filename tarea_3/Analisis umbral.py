# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:27:46 2020

@author: Pablo Saldarriaga
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
#%%
conn = sqlite3.connect('data/data_accidentes.sqlite3')

query = """ SELECT * FROM
            info"""
            
data = pd.read_sql_query(query, conn)
data['TW'] = pd.to_datetime(data['TW'])
#%%
### Agregar otras features
data['hora'] = data['TW'].dt.hour
data['dia_sem'] = data['TW'].dt.dayofweek

data= pd.get_dummies(data, columns=['hora'])
data= pd.get_dummies(data, columns=['icon'])
data= pd.get_dummies(data, columns=['dia_sem'])
#%%
### Feature augmentation
freq = '5H'
variables = ['temperature','precipIntensity','apparentTemperature','dewPoint',
             'humidity','windSpeed','cloudCover','visibility']

data_aux = data.copy()
data_aux.index = data_aux.TW
data_aux = data_aux.sort_index()
data_aux = data_aux.drop(columns = 'TW')
resample_data = data_aux[variables].rolling(freq, closed = 'left').mean()

data_pivot = data_aux.pivot_table(values=variables, index='TW',columns='BARRIO', aggfunc=sum)
data_mean = data_pivot.rolling(freq, closed = 'left').mean().stack().reset_index(drop = False)

col_means = [*data_mean.columns[:2]]
for col in data_mean.columns[2:]:
    col_means.append(col + '_mean')
    
data_mean.columns = col_means

data = data.merge(data_mean, how = 'left', on = ['TW','BARRIO'])
data = data.dropna().reset_index(drop = True)
#%%
X = data.drop("Accidente", 1).reset_index(drop=True)       # feature matrix 
y = data['Accidente'].reset_index(drop=True)               # target feature
X = X[X.columns[2:]]
#%%
### Empieza el procesamiento 

N = 100
p = 8
res = np.zeros(N,p+1)

for j in range(N):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.40)
    
    clf = ExtraTreesClassifier(n_estimators=70, random_state = 42)
    
    clf.fit(X_train, y_train)
    predu = clf.predict_proba(X_test)
    auc = roc_auc_score(y_test, predu[:,1])
    
    res[j,0] = auc
    
    for i in range(p):
        tra_0 = int(len(y_train) - y_train.sum())
        tra_1 = int(y_train.sum())
    
        prop_deseada_under = 0.1+0.05*(i+1)
        mul_updown = (tra_0 * prop_deseada_under - tra_1 * (1 - prop_deseada_under)) / (tra_0 * prop_deseada_under)   
        fac_1 = int(tra_0 * (1 - mul_updown))
    
        ratio_u = {0 : fac_1, 1 : tra_1}
        rus = RandomUnderSampler(sampling_strategy = ratio_u, random_state=42)
        X_train_i, y_train_i = rus.fit_sample(X_train, y_train)
    
    
        clf.fit(X_train_i, y_train_i)
        predu = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, predu[:,1]) 
    
        res[j,i+1] = auc
    



