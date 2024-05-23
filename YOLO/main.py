# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:28:52 2020

@author: Admin
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from load_data import LoadData
from net import Net
from visualize import drawboundingbox_1
from visualize import drawboundingbox_2
from NMS import Non_Maximum_Suppression
from NMS import non_maximum_suppression
multiplier = 200
posNum = 1 * multiplier#200
negNum = 1 * multiplier#200
train_size = posNum + negNum
test_size = 1 * multiplier#200
head = LoadData()
train_data,test_data = head.load_image(posNum,negNum,train_size,test_size)
train_label = head.load_label(posNum,train_size)
network = Net(batch = 64, lr = 1e-4, momentum = 0.9)
network.reset_grads()
epoch = 200
for j in range(0,epoch):
    # if j == 40:
    #     lr = 0.001
    # elif j == 80:
    #     lr = 0.0005
    # elif j == 140:
    #     lr = 0.0001
    
    # if j == 3:
    #     lr = 0.002
    print('\nepoch',j,':')
    for i in range(0,64):
        data = np.reshape(train_data[i],(1,3,448,448)) / 255
        result = network.predict(data)
        # normalize(data,'NORM_L2')
        # print('result',i,'= ',result)
        print('train',i,':Done!')
        network.update(i, data, train_label[i])
        train_index = np.random.choice(train_size,train_size,replace = False)
        train_data = train_data[train_index]
        
record_loss = network.return_loss()
record_params = network.return_params()

count = 0
for i in range(0,25):
    data = np.reshape(train_data[i],(1,3,448,448))
    result = network.predict(data)
    # drawboundingbox_1(train_data[i],result)
    # flag = Non_Maximum_Suppression(result)
    proposal_index = non_maximum_suppression(result)
    if proposal_index:
        drawboundingbox_2(train_data[i],proposal_index,result)
        count += 1
    print('test',i,':Done!')
print('\ncorrect:',count)
    