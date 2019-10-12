# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:34:05 2019

@author: Lenovo
"""
import numpy as np
import pandas as pd


def FitLinearKernel(X, y, lamda):
    """
    lamda is a positiveregularizer,if lamda=0 -> normal linear regression else -> ridge regression
    """
    m,n=y.shape
    I=(np.identity(m))
    alpha=np.dot(np.linalg.pinv(np.dot(X,np.transpose(X))+(lamda*I)),y)
    return alpha

def predictWithLinearKernel(X_test,alpha,X_train):
    y_pred= np.dot((np.dot(X_test,np.transpose(X_train))),alpha)
    return y_pred 