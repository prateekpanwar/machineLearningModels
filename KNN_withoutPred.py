# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

accuracies = []
for j in range(25):
#################################Data processing#######################################
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    
    df.drop(['id'], 1, inplace=True)
#################################ML#######################################    
    X = np.array(df.drop(['class'],1))
    y = np.array(df['class'])
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)
    
    clf = neighbors.KNeighborsClassifier();
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    accuracies.append(accuracy)

#################################Print the avg accuracy#######################################    
print('Final:', sum(accuracies)/len(accuracies))