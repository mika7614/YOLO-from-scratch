# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:48:34 2020

@author: Admin
"""

def coordinate_transform(src_box1,src_box2):
    boxes = [src_box1,src_box2]
    points = []
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        index = box[4]
        row = index // 7
        col = index % 7
        X = round(x * 64 + col * 64)
        Y = round(y * 64 + row * 64)
        W = round(w * 448)
        H = round(h * 448)
        points.append([Y - H // 2,X - W // 2,
                       Y + H // 2,X + W // 2])
    [dst_box1,dst_box2] = points
    return dst_box1,dst_box2