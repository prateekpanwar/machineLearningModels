# -*- coding: utf-8 -*-
#Writing the linear regression model with a big dataset
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

###############################dataset gen function####################################################
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
                val += step
        elif correlation and correlation == 'neg':
                val -= step
        xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

###################################Linear regression functions#############################
def best_fit_slope_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) / 
          ((mean(xs)**2) - mean(xs**2)) )
    b = mean(ys) - m*mean(xs)
    return m, b
    
def squared_error(ys_org, ys_line):
    return sum((ys_line - ys_org)**2)

def coff_of_determination(ys_org, ys_line):
    y_mean_line = [mean(ys_org) for y in ys_org]
    squared_error_regr = squared_error(ys_org, ys_line)
    squared_error_y_mean = squared_error(ys_org, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

####################################create a random dataset##############################	
xs, ys = create_dataset(40, 40, 2, correlation='pos')

####################################Calculating value of slope and the y intercept and implementation of the line eq##############################   
m, b = best_fit_slope_intercept(xs, ys)
print('m = ', m, 'b = ', b)
regression_line = [ (m*x)+b for x in xs]

#########################################prediction##############################################

predict_x = 8 #predict y where x=8
predict_y = (m*predict_x) + b

r_squared = coff_of_determination(ys, regression_line)
print(r_squared) #accuracy
#######################################plot######################################################
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()
