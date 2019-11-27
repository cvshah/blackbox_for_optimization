import numpy as np
from random import seed, gauss
seed(1)

#from blackboxforoptimization import Blackbox_for_Optimization
from blackbox import BFGSVK_2

class function3d(object): 
    def evaluate_function(self, x):
        f1 = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + 100 * ( x[2] - x[1] ** 2 ) ** 2 +( 1 - x[1] ) ** 2
        f2 =0.1* gauss (0,1)
        f = f1 + f2
        return f

    def gradient_function(self, x):
        g = np.array([2 * (x[0] - 1) + 400 * x[0] * ( x[0] ** 2 - x[1]), 200 * (x[1] - x[0] ** 2) + 400 * x[1] * (x[1] ** 2 - x[2]), 200 * (x[2] - x[1] ** 2)])
        return g

#problem
prob1 = function3d()
x = np.array([7,-5,-10])
BFGS = BFGSVK_2(prob1, x, line_search = True)
x___1 = BFGS[3]
x___2 = BFGS[4]
x___3 = BFGS[5]
print(BFGS[1], BFGS[6])
print(BFGS[2])
print(x___1[-1],x___2[-1], x___3[-1])
