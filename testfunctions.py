import numpy as np
import matplotlib.pyplot as plt
from random import seed, gauss

seed(1)

#from blackboxforoptimization import Blackbox_for_Optimization
from blackbox import BFGSVK_1

class function2(object):    
    def evaluate_function(self, x):
        f1 = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        f2 = gauss (0,0.00001)
        f = f1 + f2
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
'''
# Design variables at mesh points
i1 = np.linspace(-3.0, 3.0, 5000)
i2 = np.linspace(-3.0, 3.0, 5000)
x1m, x2m = np.meshgrid(i1, i2)
fm = np.zeros(x1m.shape)
for i in range(x1m.shape[0]):
    for j in range(x1m.shape[1]):
        fm[i][j] =(1 - x1m[i][j]) ** 2 + 100 * (x2m[i][j] - x1m[i][j] ** 2) ** 2 + gauss(0,1)
'''
# Create a contour plot
plt.figure()
'''
# Plot contours
CS = plt.contour(x1m, x2m, fm, [1,5,10,50,100,150,200,500,750,1000,1500,2000,5000,10000])
# Label contours
plt.clabel(CS, inline=1, fontsize=10)
print('graph is completed')
'''


# Add some text to the plot
plt.title('gauss')
plt.xlabel('x1')
plt.ylabel('x2')

prob2 = function2()
x = np.array([0,0])
BFGS = BFGSVK_1(prob2, x, line_search = True)
x___1 = BFGS[3]
x___2 = BFGS[4]
print(BFGS[1], BFGS[5])
plt.plot(x___1,x___2,'b-o')
plt.show()