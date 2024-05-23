# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:25:52 2020

@author: Admin
"""
import numpy as np
def normalize(data,mode):
    if(mode == 'max'):
        norm_factor = np.max(data)
        data = data / norm_factor
        return data
    if(mode == 'sum'):
        norm_factor = np.sum(data)
        data = data / norm_factor
        return data
    if(mode == 'NORM_L2'):
        norm_factor = np.linalg.norm(data)
        data = data / norm_factor
        return data