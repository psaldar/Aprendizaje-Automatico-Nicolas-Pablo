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
from sklearn.metrics import balanced_accuracy_score
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

cols = ['apparentTemperature', 'temperature', 'humidity_mean',
       'temperature_mean', 'apparentTemperature_mean', 'humidity',
       'dewPoint_mean', 'dewPoint', 'windSpeed_mean', 'windSpeed',
       'cloudCover_mean', 'uvIndex', 'icon_partly-cloudy-night',
       'poblado_LaAguacatala', 'visibility_mean', 'cloudCover', 'dia_sem_6',
       'precipIntensity_mean', 'visibility', 'poblado_ElCastillo',
       'icon_partly-cloudy-day', 'poblado_VillaCarlota', 'poblado_Astorga',
       'precipIntensity', 'poblado_AltosdelPoblado', 'precipProbability',
       'hora_7', 'poblado_LosBalsosNo1', 'poblado_ElDiamanteNo2',
       'poblado_Manila', 'poblado_SantaMariadeLosÁngeles', 'poblado_Lalinde',
       'hora_3', 'hora_0', 'hora_1', 'dia_sem_4', 'dia_sem_5', 'hora_2',
       'poblado_ElPoblado', 'poblado_SanLucas', 'poblado_LasLomasNo2',
       'dia_sem_1', 'dia_sem_2', 'poblado_BarrioColombia',
       'poblado_LosBalsosNo2', 'hora_19', 'hora_4', 'dia_sem_3', 'dia_sem_0',
       'hora_17', 'hora_6', 'icon_cloudy']

X = X[cols]
X_test = X_test[cols]

N = 200
p = 8
res = np.zeros((N,p+4))
res_fs = np.zeros((N,p+4))
balance_fs = np.zeros((N,p+4))

res_test = []
fscore_test = []
balance_test = []


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
        
        predu = clf.predict_proba(X_val)[:,1]
        preds = clf.predict(X_val)
        auc = roc_auc_score(y_val, predu) 
        
        if auc > best_score:
            print(auc)
            best_score = auc
            best_mod = clf
        
        res[j,i+1] = auc
        res_fs[j,i+1] = f1_score(y_val, preds)
        balance_fs[j,i+1] = balanced_accuracy_score(y_val, preds)
        
    predu_test = best_mod.predict_proba(X_test)[:,1]
    preds_test = best_mod.predict(X_test)
    
    auc_test = roc_auc_score(y_test, predu_test) 
    
    res_test.append(auc_test)
    fscore_test.append(f1_score(y_test, preds_test))
    balance_test.append(balanced_accuracy_score(y_test, preds_test))

pd.DataFrame(res).to_csv('data/sim_umbrales_roc.csv', index = False, sep = ',')
pd.DataFrame(res_test).to_csv('data/sim_umbrales_roc_test.csv', index = False, sep = ',')

pd.DataFrame(res_fs).to_csv('data/sim_umbrales_fs.csv', index = False, sep = ',')
pd.DataFrame(fscore_test).to_csv('data/sim_umbrales_fs_test.csv', index = False, sep = ',')

pd.DataFrame(balance_fs).to_csv('data/sim_umbrales_bl.csv', index = False, sep = ',')
pd.DataFrame(balance_test).to_csv('data/sim_umbrales_bl_test.csv', index = False, sep = ',')