#writing linear reg algo
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
##################################define simple dataset#####################################

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

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

####################################Calculating value of slope and the y intercept and implementation of the line eq##############################    
m, b = best_fit_slope_intercept(xs, ys)
print(m, b)
regression_line = [ (m*x)+b for x in xs]

###################################prediction##############################################

predict_x = 8 #predict y where x=8
predict_y = (m*predict_x) + b

r_squared = coff_of_determination(ys, regression_line)
print(r_squared) #accuracy

#############################plotting############################################
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
