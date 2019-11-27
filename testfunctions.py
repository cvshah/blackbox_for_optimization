import numpy as np
import matplotlib.pyplot as plt
from random import seed, gauss
import os
seed(1)

#from blackboxforoptimization import Blackbox_for_Optimization
from blackbox import BFGSVK_1

class function2(object): 
    def evaluate_function(self, x):
        f1 = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        f2 =0* gauss (0,1)
        f = f1 + f2
        return f

    def gradient_function(self, x):
        g = np.array([2 * (x[0] - 1) + 400 * x[0] * ( x[0] ** 2 - x[1]) + 0.001 * gauss(0,1), 200 * (x[1] - x[0] ** 2) + 0.001 * gauss(0,1)])
        return g
    
    def gradient_function_avg(self, x, norm_g_x_k):
        nk = 1 + abs(np.floor(np.log(norm_g_x_k)))
        print(nk)
        for g_eval in range(nk.astype(int)):
            sum_of_g = np.array([0,0], dtype= np.float64)
            sum_of_g += self.gradient_function(x)
        gradient_avg = sum_of_g / nk  
        return gradient_avg          

    def hessian_function(self, x):
        f11 = 2 + 400 * (3 * x[0] **2 - x[1])
        f12 = 400 * (-1) * x[0]
        f22 = 200
        h = np.array([[f11, f12 ], [ f12, f22]])
        return h

i1 = np.linspace(-3.0, 3.0, 5000)
i2 = np.linspace(-3.0, 3.0, 5000)
x1m, x2m = np.meshgrid(i1, i2)

# Design variables at mesh points.s
if not os.path.exists('countours.txt'):
    fm = np.zeros(x1m.shape)
    for i in range(x1m.shape[0]):
        for j in range(x1m.shape[1]):
            fm[i][j] =(1 - x1m[i][j]) ** 2 + 100 * (x2m[i][j] - x1m[i][j] ** 2) ** 2 + gauss(0,1)
    np.savetxt('countours.txt',fm)
else:
    print("Loading from existing file")
    fm = np.loadtxt('countours.txt')

# Create a contour plot
plt.figure()

# Plot contours
CS = plt.contour(x1m, x2m, fm, [ 1, 5, 10, 50, 100, 150, 200, 500, 750, 1000, 1500, 2000, 5000, 7500, 10000])
# Label contours
plt.clabel(CS, inline=1, fontsize=10)
print('graph is completed')

# the plot
plt.title('gauss')
plt.xlabel('x1')
plt.ylabel('x2')

#problem
prob1 = function2()
x = np.array([0,1])
BFGS = BFGSVK_1(prob1, x, line_search = True)
x___1 = BFGS[3]
x___2 = BFGS[4]
print(BFGS[1], BFGS[5])
plt.plot(x___1,x___2,'b-o')
print(BFGS[2])
print(x___1[-1],x___2[-1])
plt.show()
