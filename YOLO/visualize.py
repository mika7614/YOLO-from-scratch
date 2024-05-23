# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:52:37 2020

@author: Admin
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
def drawboundingbox_1(img,label):
    if img.ndim > 2:
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
    offset = 5
    for i in range(0,49):
        row = i // 7
        col = i % 7
        point_x = col * 64
        point_y = row * 64
        boundingbox_num = 2
        if np.all(label[i,0:4] == label[i,5:9]):
            if(label[i,4] == 0):
                continue
            else:
                boundingbox_num = 1
        for j in range(0,boundingbox_num):
            interval_x = round(label[i,0+j*offset] * 64)
            interval_y = round(label[i,1+j*offset] * 64)
            interval_w = round(label[i,2+j*offset] * 448 / 2)
            interval_h = round(label[i,3+j*offset] * 448 / 2)
            center_x = point_x + interval_x
            center_y =  point_y + interval_y
            left_x = center_x - interval_w
            right_x = center_x + interval_w
            left_y = center_y - interval_h
            right_y = center_y + interval_h
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img,(left_x,left_y),(right_x,right_y),(0,255,0),3)
            cv2.putText(img,'dog',(left_x,left_y),font,1,(200,100,255),2,cv2.LINE_AA)
            plt.imshow(img)
            plt.axis('off')
    plt.show()
    
def drawboundingbox_2(img,proposal_index,predict_label):
    if img.ndim > 2:
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        
    for i in proposal_index:
        true_id = i // 2
        offset = i % 2 * 5
        row = true_id // 7
        col = true_id % 7
        point_x = col * 64
        point_y = row * 64
        interval_x = round(predict_label[true_id,0+offset] * 64)
        interval_y = round(predict_label[true_id,1+offset] * 64)
        interval_w = round(predict_label[true_id,2+offset] * 448 / 2)
        interval_h = round(predict_label[true_id,3+offset] * 448 / 2)
        center_x = point_x + interval_x
        center_y =  point_y + interval_y
        left_x = center_x - interval_w
        right_x = center_x + interval_w
        left_y = center_y - interval_h
        right_y = center_y + interval_h
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img,(left_x,left_y),(right_x,right_y),(0,255,0),3)
        cv2.putText(img,'dog',(left_x,left_y),font,1,(200,100,255),2,cv2.LINE_AA)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    
    # if(np.array(group).ndim > 1):
    #     for [left_y,left_x,right_y,right_x] in group:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.rectangle(img,(left_x,left_y),(right_x,right_y),(0,255,0),3)
    #         cv2.putText(img,'dog',(left_x,left_y),font,1,(200,100,255),2,cv2.LINE_AA)
    #         plt.imshow(img)
    #         plt.axis('off')
    # else:
    #     left_y = group[0]
    #     left_x = group[1]
    #     right_y = group[2]
    #     right_x = group[3]
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.rectangle(img,(left_x,left_y),(right_x,right_y),(0,255,0),3)
    #     cv2.putText(img,'dog',(left_x,left_y),font,1,(200,100,255),2,cv2.LINE_AA)
    #     plt.imshow(img)
    #     plt.axis('off')
    # plt.show()
