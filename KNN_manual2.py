# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

###############################KNN function############################
def k_n_n(data, predict, k=3):
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_dist = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_dist, group])
    
    votes = [i[1] for i in sorted(distances) [:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    
    print(vote_result, confidence)
    return vote_result, confidence
#########################################################################################
accuracies =[]

for j in range(1):
#######################################data processing###############################
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist() #converting everything to float
    
    random.shuffle(full_data)
    
    test_size = 0.2
    train_set = {2:[], 4:[]} #defining dict class, 2 classes and 4 features
    test_set = {2:[], 4:[]} #defining dict class, 2 classes and 4 features
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]
    
    for i in train_data:
        train_set[i[-1]].append(i[:-1]) # -1 is to select the last entry in a row which is the label
    
    for i in test_data:
        test_set[i[-1]].append(i[:-1]) # -1 is to select the last entry in a row which is the label.
		
##########################################testing######################################################
    correct =0;
    total =0;
    
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_n_n(train_set, data, k=3)
            if group == vote:
                correct+=1
            total +=1
    accuracies.append(correct/total)


print(sum(accuracies)/len(accuracies))
