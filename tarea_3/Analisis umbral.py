# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:27:46 2020

@author: Pablo Saldarriaga
"""
import sqlite3
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
#%%
conn = sqlite3.connect('data/data_accidentes.sqlite3')

d_ini = dt.datetime(2017,6,1)
d_fin = dt.datetime(2018,1,1)

query = f""" SELECT * FROM
            info
            WHERE
            TW >= '{d_ini}' AND
            TW < '{d_fin}'"""
            
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
data_train = data[data['TW']<dt.datetime(2017,11,1)].reset_index(drop = True)

X = data_train.drop("Accidente", 1).reset_index(drop=True)       # feature matrix 
Y = data_train['Accidente'].reset_index(drop=True)               # target feature
X = X[X.columns[2:]]
#%%
data_test = data[(data['TW']>=dt.datetime(2017,11,1))&(data['TW']<dt.datetime(2017,11,16))].reset_index(drop = True)

X_test = data_test.drop("Accidente", 1).reset_index(drop=True)       # feature matrix 
y_test = data_test['Accidente'].reset_index(drop=True)               # target feature
X_test = X_test[X_test.columns[2:]]
#%%
### Empieza el procesamiento 

cols = ['humidity_mean', 'apparentTemperature', 'temperature', 'humidity',
       'apparentTemperature_mean', 'temperature_mean', 'dewPoint_mean',
       'icon_partly-cloudy-night', 'dewPoint', 'windSpeed_mean',
       'visibility_mean', 'uvIndex', 'windSpeed', 'cloudCover_mean', 'hora_6',
       'dia_sem_6', 'hora_0', 'cloudCover', 'precipIntensity_mean', 'hora_3',
       'icon_partly-cloudy-day', 'visibility', 'precipProbability', 'hora_7',
       'precipIntensity', 'hora_4', 'hora_1', 'hora_2']

X = X[cols]
X_test = X_test[cols]

N = 50
p = 8
res = np.zeros((N,p+4))
res_test = []



for i in range(p+3):
    
    print(f'Proporcion {0.05+0.05*(i+1-2)}')
    best_mod = None
    best_score = 0
    
    for j in range(N):        

        print(j)
        X_train_j, X_val, y_train_j, y_val = train_test_split(X, Y,
                                                        stratify=Y, 
                                                        test_size=0.25)
    
        
        clf = RandomForestClassifier(n_estimators=90, max_depth =10,
                               max_features = 'auto', bootstrap = True,
							   criterion ='entropy',n_jobs = 11)
        
        tra_0 = int(len(y_train_j) - y_train_j.sum())
        tra_1 = int(y_train_j.sum())
    
        if i>0:
            prop_deseada_under = 0.05+0.05*(i+1-2)
            mul_updown = (tra_0 * prop_deseada_under - tra_1 * (1 - prop_deseada_under)) / (tra_0 * prop_deseada_under)   
            fac_1 = int(tra_0 * (1 - mul_updown))
        
            ratio_u = {0 : fac_1, 1 : tra_1}
            rus = RandomUnderSampler(sampling_strategy = ratio_u, random_state=42)
            X_train_i, y_train_i = rus.fit_sample(X_train_j, y_train_j)
        else:
            X_train_i =X_train_j 
            y_train_i = y_train_j
    
    
        clf.fit(X_train_i, y_train_i)
        
        predu = clf.predict_proba(X_val)
        preds = clf.predict(X_val)
        auc = f1_score(y_val, preds) 
        
        if auc > best_score:
            print(auc)
            best_score = auc
            best_mod = clf
        
        res[j,i+1] = auc
        
    predu_test = best_mod.predict_proba(X_test)
    preds_test = best_mod.predict(X_test)
    
    auc_test = f1_score(y_test, preds_test) 
    res_test.append(auc_test)

pd.DataFrame(res).to_csv('data/sim_umbrales_fs.csv', index = False, sep = ',')
pd.DataFrame(res_test).to_csv('data/sim_umbrales_fs_test.csv', index = False, sep = ',')