# where I employ gradient descent on a 

import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import copy

def g(x):
    return 3*x + 2

xdata = [i for i in range(15)]
ydata = [g(x) for x in xdata]
params = [5,4]
N = len(xdata)
beta = .9

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


    theta1 = theta1 - alpha*(beta*sw + (1-beta)*sum(grad[0])/N) 
    theta2 = theta2 - alpha*(beta*sb + (1-beta)*sum(grad[1])/N)

    past_t1d.append(sum(grad[0])/N)
    past_t2d.append(sum(grad[1])/N)
    return [theta1, theta2] 

speedw = 0
speedb = 0
for i in range(0,100000):
    random.shuffle(xdata)
    xdata = copy.deepcopy(xdata)[:int(N*.2)]
    ydata = [g(x) for x in xdata]

    params = update_params(params, xdata, ydata, .01, speedw, speedb)
    speedw = sum(past_t1d)/len(past_t1d)
    speedb = sum(past_t2d)/len(past_t2d)

print(params)







# def g(x):
#     return 3*x[0] + 4*x[1]**2

# random.seed(1)
# xdata = [[random.randint(-10,10), random.randint(-10,10)] for _ in range(2)]
# ydata = [g(x) for x in xdata]
# N = len(ydata)

# params = [random.randint(-10,10), random.randint(-10,10)]

# def cost(ytrue, ypred):
#     return [(yt - yp)**2 for yt, yp in zip(ytrue, ypred)]

# def eval_grad(xdata, ytrue, params): 
#     theta1d = [-x[0]*2*(y-params[0]*x[0]-params[1]*x[1]**2) for x,y in zip(xdata,ytrue)]
#     theta2d = [-(x[1]**2)*2*(y-params[0]*x[0] - params[1]*x[1]**2) for x,y in zip(xdata,ytrue)]
#     return [theta1d, theta2d]

# def gradient_descent(xd,ytrue,params, alpha=.0001, num_iter = 100):
#     grad = eval_grad(xd,ytrue,params)
#     for _ in range(num_iter):
#         new_theta1 = params[0] - (1/N)*(alpha)*sum(grad[0])
#         new_theta2 = params[1] - (1/N)*(alpha)*sum(grad[1])
#         new_params = [new_theta1, new_theta2]
#         grad = eval_grad(xd,ytrue,new_params)

#     return new_params

# gradient_descent(xdata, ydata, params)