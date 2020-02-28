# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:47:11 2020

@author: pasal
"""
import sqlite3
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
#%%
conn = sqlite3.connect('data/data_accidentes.sqlite3')

d_ini = dt.datetime(2017,6,1)
d_fin = dt.datetime(2017,11,1)

query = f""" SELECT * FROM
            info
            WHERE
            TW >= '{d_ini}' AND
            TW < '{d_fin}'"""
            
data = pd.read_sql_query(query, conn)
data['TW'] = pd.to_datetime(data['TW'])
#%%
### Dejar solo accidentes del Poblado
lista_poblado = """
ElGuamal
BarrioColombia
VillaCarlota
Castropol
Lalinde
LasLomasNo1
LasLomasNo2
AltosdelPoblado
ElTesoro
LosNaranjos
LosBalsosNo1
SanLucas
ElDiamanteNo2
ElCastillo
LosBalsosNo2
Alejandria
LaFlorida
ElPoblado
Manila
Astorga
PatioBonito
LaAguacatala
SantaMariadeLosÁngeles
"""
lista_poblado_l = lista_poblado.split('\n')

data = data[data['BARRIO'].isin(lista_poblado_l)]
data['poblado'] = data['BARRIO']
data= pd.get_dummies(data, columns=['poblado'])
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
N = 100
res = np.zeros((N,X.shape[1]))

for i in range(N):
    print(i)
    clf = RandomForestClassifier(n_estimators=90, max_depth =10,
                               max_features = 'auto', bootstrap = True,
							   criterion ='entropy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.2)
    
    tra_0 = int(len(y_train) - y_train.sum())
    tra_1 = int(y_train.sum())
    
    prop_deseada_under = 0.5
    mul_updown = (tra_0 * prop_deseada_under - tra_1 * (1 - prop_deseada_under)) / (tra_0 * prop_deseada_under)   
    fac_1 = int(tra_0 * (1 - mul_updown))
    
    ratio_u = {0 : fac_1, 1 : tra_1}
    rus = RandomUnderSampler(sampling_strategy = ratio_u)
    X_train_under, y_train_under = rus.fit_sample(X_train, y_train)    
    
    clf.fit(X_train_under, y_train_under)
    
    res[i,:] = clf.feature_importances_
    
    pd.DataFrame(res, columns = X_train.columns).to_csv('data/feature_importance3.csv', sep =',', index = 'False')
