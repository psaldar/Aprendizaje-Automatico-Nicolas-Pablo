# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:58:47 2020

@author: nicol
"""


import sqlite3
import pandas as pd
import datetime as dt
import numpy as np
import random




### Silencio warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


## Fijo semilla aleatoria 
import tensorflow as tf
import os

SEED = 1  # use this constant seed everywhere

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)  # `python` built-in pseudo-random generator
np.random.seed(SEED)  # numpy pseudo-random generator
tf.set_random_seed(SEED)  # tensorflow pseudo-random generator




### Lectura del dataset a usar (se usa un subconjunto del dataset por motivos de tiempo)

d_ini = dt.datetime(2017,6,1)
d_fin = dt.datetime(2017,7,1)
db_name = r"C:\Users\nicol\Documents\Mis_documentos\MaestriaCienciaDatos\Semestre2\AprendizajeMaq\Github\Apr\tarea_1\data\data_accidentes.sqlite3"

conn = sqlite3.connect(db_name)
query = f"""SELECT *
            FROM info
            WHERE
            TW >= '{d_ini}' AND
            TW <= '{d_fin}'  """
data = pd.read_sql_query(query,conn)
data['TW'] = pd.to_datetime(data['TW'])


### Temporal, dejar solo algunos barrios
data = data[data['BARRIO'].str.contains("LaA")]






############### Agregar otras features
data['mes'] = data['TW'].dt.month
data['dia_sem'] = data['TW'].dt.dayofweek

data= pd.get_dummies(data, columns=['icon'])
data= pd.get_dummies(data, columns=['dia_sem'])




















###### Seleccionar variable entrada y salida 
X = data.drop("Accidente", 1).reset_index(drop=True)       # feature matrix 
y = data['Accidente'].reset_index(drop=True)               # target feature
X = X[X.columns[2:]]


### Partir en train y test los datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25,  random_state = 42)




### Hacer undersampling (los datos estan desbalanceados)
from imblearn.under_sampling import RandomUnderSampler

tra_0 = int(len(y_train) - y_train.sum())
tra_1 = int(y_train.sum())

prop_deseada_under = 0.05
mul_updown = (tra_0 * prop_deseada_under - tra_1 * (1 - prop_deseada_under)) / (tra_0 * prop_deseada_under)   
fac_1 = int(tra_0 * (1 - mul_updown))

ratio_u = {0 : fac_1, 1 : tra_1}
rus = RandomUnderSampler(sampling_strategy = ratio_u, random_state=42)
X_train, y_train = rus.fit_sample(X_train, y_train)





###################### FEATURE SELECTION ####################################

#### Usaremos metodos de envoltura (wrapper) para seleccionar variables.
### Realizaremos una comparacion entre una implementacion propia de forward 
### selection, y las versiones de forward y backward selection de mlxtend

### en el metodo propio usamos un conjunto de validation (holdout) para evaluar
### cada combinacion, mientras que en los de mlxtend se usa 2-fold cross validation
### (sabemos que esto puede generar diferencias entre el propio y el de mlxtend)




######### El modelo base que usaremos es un RandomForest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=35,  random_state = 1)









############### Primero, haremos un forward selection con nuestro propio codigo

print('\nSeleccionando variables con forward selection propio...')
### la metrica que usaremos es roc_auc
from sklearn.metrics import roc_auc_score
metrica_usada = roc_auc_score

### Partir los de train en validation y train
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,
                                                    stratify=y_train, 
                                                    test_size=0.25, random_state = 42)


### Ahora si, el codigo para seleccionar variables forward
elegids = []
metri_it = []
vars_it = []
features = list(X.columns)
eleg = []

### Itero por todas las columnas
for va in range(len(features)):
    ### Restan por agregar
    restantes = list(set(features) - set(eleg))
    ### Reinicio la metrica
    best_metri = 0
    ### En cada ciclo, miro todas las columnas restantes
    for restan in restantes:    
        ### Agrego temporalmente variable
        eleg_temp = eleg.copy()
        eleg_temp.append(restan)    
        ### Estimo el modelo con esas variables
        clf.fit(X_tr[eleg_temp], y_tr)    
        ### Predigo para test
        predi = clf.predict(X_val[eleg_temp])    
        ### Evaluo metrica
        metri = metrica_usada(y_val, predi)    
        ### Si mejoro a la anterior + un epsilon, actualizo
        if metri > best_metri + 0.000001:
            best_eleg = eleg_temp.copy()
            best_metri = metri
    
    ### Agrego la mejor variable de esta iteracion
    eleg = best_eleg.copy()
    ### Guardo las mejores metricas de cada iteracion
    metri_it.append(best_metri.copy())
    vars_it.append(eleg.copy())
    
print('\nMejores variables con forward selection metodo propio :')
print(vars_it[np.argmax(metri_it)])






#############  Ahora, evaluemos forward selection Con mlxtend
print('\nSeleccionando variables con forward selection de mlxtend...')

import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.evaluate import PredefinedHoldoutSplit





# Build step forward feature selection
sfs1 = sfs(clf,
           k_features='best',
           forward=True,
           floating=False,
           verbose=0,
           scoring='roc_auc',
           cv=2)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)

# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print('\nMejores variables con forward selection de mlxtend :')
print(list(X.columns[feat_cols]))

### Evolucion a traves del tiempo
evol_1 = sfs1.subsets_





###########  Por ultimo, evaluemos forward selection Con mlxtend
print('\nSeleccionando variables con backward selection de mlxtend...')

import pandas as pd

# Build step forward feature selection
sfs2 = sfs(clf,
           k_features='best',
           forward=False,
           floating=False,
           verbose=0,
           scoring='roc_auc',
           cv=2)

# Perform SFFS
sfs2 = sfs2.fit(X_train.values, y_train.values)

# Which features?
feat_cols = list(sfs2.k_feature_idx_)
print('\nMejores variables con backward selection de mlxtend :')
print(list(X.columns[feat_cols]))


### Print

### Evolucion a traves del tiempo
evol_2 = sfs2.subsets_
