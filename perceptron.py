# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 04:14:10 2019

@author: Lenovo
"""

import numpy as np
from matplotlib import pyplot as plt


#x1,x2,b
X= np.array([[0.08, 0.72, 1.0],
     [0.10, 1.00, 1.0],
     [0.26, 0.58, 1.0],
     [0.35, 0.95, 1.0],
     [0.45, 0.15, 1.0],
     [0.60, 0.30, 1.0],
     [0.70, 0.65, 1.0],
     [0.92, 0.45, 1.0]])
Y=np.array([1,-1,1,-1,1,1,-1,-1])


#y=l1
def perceptron(X,Y):
    eta=1
    w=np.zeros(len(X[0]))
    print(w)
    while(True):
        count=0
        for i,x in enumerate(X):
            if (np.dot(w,X[i])*Y[i]) <= 0:
               w = w + eta*X[i]*Y[i]
               count=count+1
        if count==0:
            print(w)
            break
    return w    
def predict(X,W):
    if np.dot(X, W) >=0.5:
        label =1
    else:
        label=0
    return label         

if __name__ == '__main__':        
    w=perceptron(X,Y)
    plot_x = np.array([np.min(X[:, 0] - 0.2), np.max(X[:, 1]+0.2)])
    plot_y = - 1 / w[1] * (w[0] * plot_x + w[2])  # comes from, w0*x + w1*y + b = 0  then y = (-1/w1) (w0*x + b)

    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, cmap='viridis')
    plt.plot(plot_x, plot_y, color='k', linewidth=2)
    plt.xlim([-0.2, 1.2]); plt.ylim([-0.2, 1.25]);
    plt.show()

    print(predict([1,1],w)
















