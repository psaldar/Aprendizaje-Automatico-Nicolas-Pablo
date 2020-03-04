# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:25:00 2020

@author: nicol
"""


##### Este archivo contiene todos los scripts utilitarios usados 
##### para el algoritmo genetico

import numpy as np
from sklearn.metrics import balanced_accuracy_score
import random
random.seed(42)
np.random.seed(42)


### Elegir solo estas features
def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features



### Accuracy
def classification_accuracy(labels, predictions):
    metric = balanced_accuracy_score(labels, predictions)
    return metric


### Calcular evaluaciones
def cal_pop_fitness(pop, features, labels, train_indices, test_indices, classifier, ya_vistas):
    accuracies = np.zeros(pop.shape[0])
    idx = 0

    for curr_solution in pop:
        reduced_features = reduce_features(curr_solution, features)
        train_data = reduced_features[train_indices, :]
        test_data = reduced_features[test_indices, :]

        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        
        
        ### Si ya se habia visto antes esta solucion
        if str(curr_solution) in ya_vistas:
            accuracies[idx] = ya_vistas[str(curr_solution)]
        
        ### de lo contrario, estimar valor de funcion objetivo
        else:
            classifier.fit(X=train_data, y=train_labels)
    
            predictions = classifier.predict(test_data)
            accuracies[idx] = classification_accuracy(test_labels, predictions)
            
            ya_vistas[str(curr_solution)] = accuracies[idx]
        
            
        idx = idx + 1
    return accuracies, ya_vistas


#### Seleccionar los padres, por torneo entre 3
def select_mating_pool(pop, fitness, num_parents):
    # Seleccionar los mejores padres, usando torneo entre 3    
    parents = np.empty((num_parents, pop.shape[1]))
    
    ### Padres no elegidos
    no_eleg = list(range(len(fitness)))
    

    ### Dejar siempre el mejor padre
    for parent_num in range(1):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
        
        ### Quitar el padre de los permitidos
        no_eleg.remove(max_fitness_idx)        
        
    
    ### Elegir el resto padres por torneo
    for parent_num in range(1,num_parents):
        parents_torneo = random.sample(no_eleg, 3)
        
        ### elegir el mejor de estos
        mejor_padre = None
        mejor_val = 0

        for pa in parents_torneo:
            if fitness[pa]>mejor_val:
                mejor_val = fitness[pa]
                mejor_padre = pa
        
        parents[parent_num, :] = pop[mejor_padre, :]
        fitness[mejor_padre] = -99999999999     
        
        ### Quitar el padre de los permitidos
        no_eleg.remove(mejor_padre)
        

     

    return parents


### Hacer el cruce
### Por ahora, es un simple one point crossover
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    
    # Punto donde se hace el cruce.
    # Elegir aleatorio el punto de corte, pero al menos 10 genes de cada padre
    crossover_point = np.random.randint(10, int(offspring_size[1])-10)

    for k in range(offspring_size[0]):
        # Indice del primer padre
        parent1_idx = k%parents.shape[0]
        # Indice del segundo padre
        parent2_idx = (k+1)%parents.shape[0]
        # La primera mitad de los genes es del primer padre
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # La segunda mitad de los genes es del segundo padre
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring





#### Esto varia genes del hijo, de forma aleatoria (cambia un 1 por un 0)
def mutation(offspring_crossover, num_mutations, num_genes):
    for i in range(num_genes):
        mutation_idx = np.random.randint(low=0, high=offspring_crossover.shape[1], size=num_mutations)
        # La mutacion cambia genes de forma aleatoria (1 por 0 o viceversa)
        for idx in range(offspring_crossover.shape[0]):
            offspring_crossover[idx, mutation_idx] = 1 - offspring_crossover[idx, mutation_idx]
    return offspring_crossover