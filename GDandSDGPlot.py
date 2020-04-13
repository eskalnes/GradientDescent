import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from mpl_toolkits.mplot3d import Axes3D
import copy

def g(x):
    return 3 * x + 2

xdata = np.linspace(0, 8,  100)

ydata = [g(x) for x in xdata]
params = [5,4]
N = len(ydata)

def eval_grad(p,xd,yt):
    theta1_d = [-x*2*(y-p[0]*x-p[1]) for x, y in zip(xd,yt)]
    theta2_d = [-2*(y-p[0]*x-p[1]) for x,y in zip(xd,yt)]
    return [theta1_d, theta2_d]

def update_params(p,xd,yt,alpha):
    grad = eval_grad(p,xd,yt)
    theta1 = p[0]
    theta2 = p[1]
    theta1 = theta1 - (alpha/N)*sum(grad[0])
    theta2 = theta2 - (alpha/N)*sum(grad[1])
    return [theta1, theta2] 

weights = []
for i in range(0,1000):
    params = update_params(params, xdata, ydata, .01)
    weights.append([params[1], params[0]])

def cost(xd, yt, params):
    return ((yt - (np.array(xd) * params[1] + params[0]))**2).sum()

def plot_counter(xdata, ydata, alpha, N, weights):

    w_history = np.array([(alpha/N)*cost(xdata, ydata, w) for w in weights])

    #Setup of meshgrid of theta values
    T0, T1 = np.meshgrid(np.linspace(-1,5,200),np.linspace(-1,5,200))

    #Computing the cost function for each theta combination
    zs = np.array([(alpha/N) * cost(xdata, ydata, [t0,t1]) for t0, t1 in zip(np.ravel(T0), np.ravel(T1))])

    Z = zs.reshape(T0.shape)

    w0 = np.array(weights)[:,0]
    w1 = np.array(weights)[:,1]

    #Angles needed for quiver plot
    anglesx = w0[1:] - w0[:-1]
    anglesy = w1[1:] - w1[:-1]

    %matplotlib inline
    fig = plt.figure(figsize = (16,8))

    #Surface plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(T0, T1, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
    ax.plot(w0,w1,w_history, marker = '*', color = 'r', alpha = .4, label = 'Gradient descent')

    ax.set_xlabel('theta 0')
    ax.set_ylabel('theta 1')
    ax.set_zlabel('Cost function')

    ax.view_init(45, 45)

    # #Contour plot
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(T0, T1, Z, 70, cmap = 'jet')
    ax.quiver(w0[:-1], w1[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .9)
    
    plt.show()
    return w_history

w_hist = plot_counter(xdata, ydata, 0.1, N, weights)

def g(x):
    return 3*x + 2

xdata = np.linspace(0, 8,  100)

ydata = [g(x) for x in xdata]
params = [5,4]
N = len(ydata)

def eval_grad(p,xd,yt):
    theta1_d = [-x*2*(y-p[0]*x-p[1]) for x, y in zip(xd,yt)]
    theta2_d = [-2*(y-p[0]*x-p[1]) for x,y in zip(xd,yt)]
    return [theta1_d, theta2_d]

def update_params(p,xd,yt,alpha):
    grad = eval_grad(p,xd,yt)
    theta1 = p[0]
    theta2 = p[1]
    theta1 = theta1 - (alpha/N)*sum(grad[0])
    theta2 = theta2 - (alpha/N)*sum(grad[1])
    return [theta1, theta2] 

weights = []
params = [5,4]
N = len(ydata)
data = copy.deepcopy(xdata)
for i in range(0,100):
    random.shuffle(data)
    data = copy.deepcopy(data)[:int(N*.2)]
    y = [g(x) for x in data]
    params = update_params(params, data, y, .01)
    weights.append([params[1], params[0]])

w_hist = plot_counter(xdata, ydata, .01, N, weights)