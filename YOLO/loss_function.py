import numpy as np

def yolo_loss(train_array, label_array):
    lambda_coord = 5
    lambda_noobj = 0.5
    loss = 0
    for i in range(label_array.shape[0]):
        if label_array[i][4] != 0:
            loss += lambda_coord * ((train_array[i][0] - label_array[i][0]) ** 2 + (train_array[i][1] - label_array[i][1]) ** 2)
            loss += lambda_coord * ((train_array[i][2] - label_array[i][2])**2 + (train_array[i][3] - label_array[i][3])**2)
            loss += (train_array[i][4] - label_array[i][4]) ** 2
            loss += lambda_coord * ((train_array[i][5] - label_array[i][5]) ** 2 + (train_array[i][6] - label_array[i][6]) ** 2)
            loss += lambda_coord * ((train_array[i][7] - label_array[i][7])**2 + (train_array[i][8] - label_array[i][8])**2)
            loss += (train_array[i][9] - label_array[i][9]) ** 2
            loss += (train_array[i][10] - label_array[i][10]) ** 2
        else:
            loss += lambda_noobj * (train_array[i][4] - label_array[i][4]) ** 2
            loss += lambda_noobj * (train_array[i][9] - label_array[i][9]) ** 2

    return loss



def yolo_return_loss(train_array, label_array):
    #train_array为神经网络输出数据，label_array为标签
    lambda_coord = 5
    lambda_noobj = 0.5
    out_array = np.zeros_like(train_array)
    
    for i in range (label_array.shape[0]):
        if label_array[i][4] != 0:
            # 第一个bounding box
            out_array[i][0] = 2 * lambda_coord * (train_array[i][0] - label_array[i][0])
            out_array[i][1] = 2 * lambda_coord * (train_array[i][1] - label_array[i][1])
            out_array[i][2] = lambda_coord * (train_array[i][2] - label_array[i][2]) * 2
            out_array[i][3] = lambda_coord * (train_array[i][3] - label_array[i][3]) * 2
            out_array[i][4] = 2 * (train_array[i][4] - label_array[i][4])  #confident

            # 第二个bounding box
            out_array[i][5] = 2 * lambda_coord * (train_array[i][5] - label_array[i][5])
            out_array[i][6] = 2 * lambda_coord * (train_array[i][6] - label_array[i][6])
            out_array[i][7] = lambda_coord * (train_array[i][7] - label_array[i][7]) * 2
            out_array[i][8] = lambda_coord * (train_array[i][8] - label_array[i][8]) * 2
            out_array[i][9] = 2 * (train_array[i][9] - label_array[i][9])   #confident
            out_array[i][10] = 2 * (train_array[i][10] - label_array[i][10])
        else:
            # 没有object的情况
            out_array[i][4] = 2 * lambda_noobj * (train_array[i][4] - label_array[i][4])
            out_array[i][9] = 2 * lambda_noobj * (train_array[i][9] - label_array[i][9])
    
    return out_array
