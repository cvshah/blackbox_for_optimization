import numpy as np
import matplotlib.pyplot as plt

#from blackboxforoptimization import Blackbox_for_Optimization
from blackbox import *
class function1(object):

    def evaluate_function(self, x):

        f = np.sum(x ** 4)
        
        return f

    def gradient_function(self, x):
        
        g = 4 * x ** 3
        return g

    def hessian_function(self, x):

        h = np.diag(12 * x **2)
        return h

class function2(object):
    
    def evaluate_function(self, x):

        f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        return f

    def gradient_function(self, x):

        g = np.array([2 * (x[0] - 1) + 400 * x[0] * ( x[0] ** 2 - x[1]), 200 * (x[1] - x[0] ** 2)])

        return g

    def hessian_function(self, x):
        f11 = 2 + 400 * (3 * x[0] **2 - x[1])
        f12 = 400 * (-1) * x[0]
        f22 = 200

        h = np.array([[f11, f12 ], [ f12, f22]])

        return h


prob1 = function1()
newtonvalue = np.zeros(200)
for i in range(1, 201):
    x = 0.1 * np.ones(i)
    newton_result = Newton(prob1, x, line_search = False)
    newtonvalue[i-1] = newton_result[1]
plt.semilogy(newtonvalue)
plt.show()
