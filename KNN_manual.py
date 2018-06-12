# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

################dataset######################
dataset = {'k': [ [1,2], [2,3], [3,1] ], 'r': [[6,5], [7,7], [8,6]]}
new_features = [1,3]

####################KNN function#####################
def k_n_n(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('Groups cant be more than data length')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_dist = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_dist, group])
    
    votes = [i[1] for i in sorted(distances) [:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

#################ML###################################
results = k_n_n(dataset, new_features, k=3)
print(results)
#######################plot#############################
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], color=results)
plt.show()