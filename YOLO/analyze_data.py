# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:57:19 2020

@author: Admin
"""
import numpy as np
from scipy.linalg import svd
def PCA(data,k):
    n = data.shape[0]
    data = data / np.sqrt(n - 1)
    (U,S,V) = svd(data,full_matrices = False)
    V_k = V[0:k,:]
    outData = np.dot(data,V_k.T)
    # for i in range(0,k):
        # print('var',np.var(outData[:,i]))
    # print('S =',S)
    return outData

def 