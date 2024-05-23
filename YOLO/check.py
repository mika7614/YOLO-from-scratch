# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:30:39 2020

@author: Admin
"""
import numpy as np
def numerical_gradient_test(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    i = 0
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        if i == 15:
            break
        i += 1
        it.iternext()   
        
    return grad

# X = np.ones((1,3))
# def f(X):
#     W = np.random.randn(3,1)
#     b = 0
#     Y = np.dot(X,W) + b
#     return Y
    
# print(numerical_gradient(f,X))