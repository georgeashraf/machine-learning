# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:29:55 2019

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data[:, :2]  # consider only 2 features
Y = (iris.target != 0) * 1  # only 2 classes
#print((Y))
b = np.ones((X.shape[0], 1))
X=np.concatenate((X,b), axis=1)

'''plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='b', label='0')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='r', label='1')
plt.legend()'''
def sigmoid(z):
    final_result = 1/(1+np.exp(-z))
    return final_result


def fit(X,Y,lr,no_iter):
    w = np.zeros(X.shape[1])
    for i in range(no_iter):
        z=sigmoid(np.dot(X,w))
        loss=(-Y * np.log(z) - (1 - Y) * np.log(1 - z)).mean()
        gradient = np.dot(X.T, (z - Y)) / Y.size
        w -=lr * gradient
        if( i % 1000 == 0):
                print(f'loss: {loss} \t')
    return w           

def predict(X,W): 
    if sigmoid(np.dot(X, W)) >=0.5:
        label =1
    else:
        label=0
    return label    
if __name__ == '__main__':
    w=fit(X,Y,0.1,30000) 
    x1_min, x1_max = X[:,0].min(), X[:,0].max()
    x2_min, x2_max = X[:,1].min(), X[:,1].max()      
    plot_x = np.array([ x1_min, x1_max])
    plot_y = - 1 / w[1] * (w[0] * plot_x + w[2])  # comes from, w0*x + w1*y + b = 0  then y = (-1/w1) (w0*x + b)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, cmap='viridis')
    plt.plot(plot_x, plot_y, color='k', linewidth=2)

    plt.xlim([x1_min, x1_max+0.2]); plt.ylim([x2_min-0.2,x2_max+0.2]);
    plt.show()
    print(predict([4.5,4.5,1],w))
        