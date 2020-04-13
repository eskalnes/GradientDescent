# where I employ gradient descent on a 

import numpy as np
import matplotlib.pyplot as plt
import sys
import random

def g(x):
    return 3*x + 2

xdata = [i for i in range(8)]

ydata = [g(x) for x in xdata]
params = [5,4]
N = len(ydata)

def eval_grad(p,xd,yt):
    theta1_d = [-x*2*(y-p[0]*x-p[1]) for x, y in zip(xd,yt)]
    theta2_d = [-2*(y-p[0]*x-p[1]) for x,y in zip(xd,yt)]
    return [theta1_d, theta2_d]



def update_params(p,xd,yt,alpha, eps):
    global g1
    global g2
    grad = eval_grad(p,xd,yt)
    theta1 = p[0]
    theta2 = p[1]
    
    # we keep a g1 and g2 to dynamcially update lr (alpha/N) 
    theta1 = theta1 - ((alpha/N)/ np.sqrt(g1 + eps))*sum(grad[0])
    theta2 = theta2 - ((alpha/N)/ np.sqrt(g2 + eps))*sum(grad[1])
    g1 += (sum(grad[0]) ** 2)
    g2 += (sum(grad[1]) ** 2)
                       
                       
    return [theta1, theta2] 

eps = 1
g1 = 0
g2 = 0
for i in range(0,1000):
    params = update_params(params, xdata, ydata, 0.1, eps)

print(params)
