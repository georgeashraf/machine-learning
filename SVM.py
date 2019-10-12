# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 07:35:54 2019

@author: Lenovo
"""
import numpy as np
from matplotlib import pyplot as plt
X= np.array([[0.08, 0.72, 1.0],
     [0.10, 1.00, 1.0],
     [0.26, 0.58, 1.0],
     [0.35, 0.95, 1.0],
     [0.45, 0.15, 1.0],
     [0.60, 0.30, 1.0],
     [0.70, 0.65, 1.0],
     [0.92, 0.45, 1.0]])
#y=1
Y=np.array([1,-1,1,-1,1,1,-1,-1])

def SVM(X,Y,epochs,lr):
    w = np.zeros(len(X[0]))
    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            if (np.dot(w,X[i])*Y[i]) < 1:
                w = w + lr * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
            else:
                w = w + lr * (-2  *(1/epoch)* w)                
    return w

def predict(X,W):
    result = np.sign(np.dot(X,W))
    return result

if __name__ == '__main__':        
    w=SVM(X,Y,10000,0.1)
    plot_x = np.array([np.min(X[:, 0] - 0.5), np.max(X[:, 1]+0.2)])
    plot_y = - 1 / w[1] * (w[0] * plot_x + w[2])  # comes from, w0*x + w1*y + b = 0  then y = (-1/w1) (w0*x + b)

    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, cmap='viridis')
    plt.plot(plot_x, plot_y, color='k', linewidth=2)
    plt.xlim([-0.2, 1.2]); plt.ylim([-0.2, 1.25]);
    plt.show()        
    print(predict([0.6,0.2,1],w))        
            
            
            
            
            
            
            
            
    