# where I employ gradient descent on a 

import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import copy

def g(x):
    return 3*x + 2

xdata = [i for i in range(20)]
ydata = [g(x) for x in xdata]
params = [5,4]
N = len(xdata)
beta = .9
eps = .0001

past_t1d = []
past_t2d = []

def eval_grad(p,xd,yt):
    theta1_d = [-x*2*(y-p[0]*x-p[1]) for x, y in zip(xd,yt)]
    theta2_d = [-2*(y-p[0]*x-p[1]) for x,y in zip(xd,yt)]

    return [theta1_d, theta2_d]


def update_params(p,xd,yt,alpha, sw, sb):
    grad = eval_grad(p,xd,yt)
    theta1 = p[0]
    theta2 = p[1]

    theta1 = theta1 - (alpha/np.sqrt(abs(sw)+eps))*(beta*sw + (1-beta)*sum([g**2 for g in grad[0]])/N) 
    theta2 = theta2 - (alpha/np.sqrt(abs(sb)+eps))*(beta*sb + (1-beta)*sum([g**2 for g in grad[1]])/N)

    # theta1 = theta1 - alpha*(beta*sw + (1-beta)*sum(grad[0])/N) 
    # theta2 = theta2 - alpha*(beta*sb + (1-beta)*sum(grad[1])/N)

    past_t1d.append(sum([g**2 for g in grad[0]])/N)
    past_t2d.append(sum([g**2 for g in grad[1]])/N)
    return [theta1, theta2] 

speedw = 0
speedb = 0
for i in range(0,10):
    random.shuffle(xdata)
    xdata = copy.deepcopy(xdata)[:int(N*.2)]
    ydata = [g(x) for x in xdata]

    params = update_params(params, xdata, ydata, .01, speedw, speedb)
    speedw = sum(past_t1d)/len(past_t1d)
    speedb = sum(past_t2d)/len(past_t2d)
    print(params)


print(params)







