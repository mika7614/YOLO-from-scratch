import numpy as np
from coordinate_transform import coordinate_transform
def iou(box1, box2):
    '''
    两个框（二维）的 iou 计算
    注意：边框以左上为原点
    box:[top, left, bottom, right]    [y1, x1, y2, x2]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    if union == 0:
        print(box1,box2)
        return -1
    iou = inter / union
    return iou

def Non_Maximum_Suppression(predict_array):  #predict为预测结果（49，11）
    index = 2
    predict_array = predict_array.copy()
    # index_array = np.array((0, predict_array[0][4] * predict_array[0][10]))  #index_array 存储候选框的索引和confident*概率
    index_array = np.array((0, predict_array[0][4]))
    # temp = np.array((1, predict_array[0][9] * predict_array[0][10]))
    temp = np.array((1, predict_array[0][9]))
    index_array = np.vstack((index_array,temp))
    for i in range(0,predict_array.shape[0] - 1):
        # temp = np.array((index, predict_array[i][4] * predict_array[i][10]))
        temp = np.array((index, predict_array[i][4]))
        index_array = np.vstack((index_array,temp))
        index += 1
        # temp = np.array((index, predict_array[i][9] * predict_array[i][10]))
        temp = np.array((index, predict_array[i][9]))
        index_array = np.vstack((index_array, temp))
        index += 1
        i += 1
    sort_array = np.argsort(-index_array[:, 1])  #得到confident的排序索引（从小到大）
    remain_array = [x for x in index_array[sort_array] if x[1] > 1]   #保留confident大于0.2的框
    remain_array = list(np.array(remain_array).astype(np.int))
    if remain_array:
        final_array = [remain_array[0]]  #数组，final_array存储每次IOU最大的候选框
        max_bounding = []
    output_array = []      #存储
    while remain_array :
        del remain_array[0]
        
        max_confidence_id = final_array[-1][0] // 2 #取整，确定哪一行
        boundingbox_id = final_array[-1][0] % 2 #取余，确定第几个框
        if boundingbox_id == 0:
            x = predict_array[max_confidence_id][0]
            y = predict_array[max_confidence_id][1]
            w = predict_array[max_confidence_id][2]
            h = predict_array[max_confidence_id][3]
        else:
            x = predict_array[max_confidence_id][5]
            y = predict_array[max_confidence_id][6]
            w = predict_array[max_confidence_id][7]
            h = predict_array[max_confidence_id][8]
            
        for i in range(len(remain_array)):
            proposal_id = remain_array[i][0] // 2   #待比较的候选框
            boundingbox_id = remain_array[i][0] % 2
            if boundingbox_id == 0:
                xx = predict_array[proposal_id][0]
                yy = predict_array[proposal_id][1]
                ww = predict_array[proposal_id][2]
                hh = predict_array[proposal_id][3]
            else:
                xx = predict_array[proposal_id][5]
                yy = predict_array[proposal_id][6]
                ww = predict_array[proposal_id][7]
                hh = predict_array[proposal_id][8]
            max_bounding = [x, y, w, h, max_confidence_id]  #confident最大候选框的坐标
            bounding = [xx, yy, ww, hh, proposal_id]  #IOU计算的另一个候选框坐标
            max_bounding, bounding = coordinate_transform(max_bounding, bounding)  #坐相对标转化为绝对坐标
            # print(max_bounding,bounding)
            remain_array[i][1] = iou(max_bounding, bounding)
        output_array.append(max_bounding)  #存储每次最大bounding box的坐标
        remain_array = np.array(remain_array)
        dele = remain_array > 0.8
        # print(dele.shape,dele.ndim)
        if dele.ndim > 1:
            mask = dele[:, 1]
        else:
            continue
        remain_array = np.delete(remain_array, mask, 0)
        remain_array = list(remain_array)
        if remain_array:
            final_array.append(remain_array[0])
    return output_array  #返回NMS剩下候选框坐标

def non_maximum_suppression(predict_array):
    predict_array = predict_array.copy()
    index_array = []
    confidence_threshold = 0.6
    j = 0
    for i in range(0,predict_array.shape[0]):
        index_array.append(np.array([j, predict_array[i][4]]))
        index_array.append(np.array([j+1, predict_array[i][9]]))
        j += 2
    index_array = np.array(index_array)
    max_confidence = np.max(index_array[:,1])
    if max_confidence > 1:
        index_array[:,1] = index_array[:,1] / max_confidence
    sort_array_index = np.argsort(-index_array[:, 1])
    index_array = index_array[sort_array_index]
    
    mask = index_array[:,1] > confidence_threshold
    
    remain_array = index_array[mask].astype(np.int)
    output_index = []
    while remain_array.size > 0:
        output_index.append(remain_array[0,0])
        remain_array = np.delete(remain_array,0,0)
        
        max_confidence_id = output_index[-1] // 2
        boundingbox_id = output_index[-1] % 2
        offset = 5 * boundingbox_id
        x = predict_array[max_confidence_id][0 + offset]
        y = predict_array[max_confidence_id][1 + offset]
        w = predict_array[max_confidence_id][2 + offset]
        h = predict_array[max_confidence_id][3 + offset]
            
        for i in range(0,remain_array.shape[0]):
            proposal_id = remain_array[i][0] // 2
            boundingbox_id = remain_array[i][0] % 2
            offset = 5 * boundingbox_id
            xx = predict_array[proposal_id][0 + offset]
            yy = predict_array[proposal_id][1 + offset]
            ww = predict_array[proposal_id][2 + offset]
            hh = predict_array[proposal_id][3 + offset]
            
            max_bounding = [x, y, w, h, max_confidence_id]
            bounding = [xx, yy, ww, hh, proposal_id]
            max_bounding, bounding = coordinate_transform(max_bounding, bounding)
            remain_array[i][1] = iou(max_bounding, bounding)
        mask = remain_array[:,1] > 0.7
        remain_array = np.delete(remain_array, mask, 0)
    return output_index

