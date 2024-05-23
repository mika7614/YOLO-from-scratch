# -*- coding: utf-8 -*-
import numpy as np
import cv2
from xml.dom.minidom import parse
MODE = cv2.IMREAD_COLOR
# cv2.IMREAD_COLOR
# cv2.IMREAD_GRAYSCALE
class LoadData:
    def __init__(self):
        self.train_data_ID = []
        self.test_data_ID = []
        # self.trainIndex = None
        # self.testIndex = None
        
    def load_image(self,posNum,negNum,trainSize,testSize):
        train_data = []
        offset = 0
        
        # ******************* load train_data *****************#
        #parse postive train_data's ID
        offset = self.readDataID('dog_train.txt', posNum, offset, self.train_data_ID)
        
        for i in range(0,posNum):
            filename = 'VOCtrainval_06-Nov-2007\\VOCdevkit\\'\
                'VOC2007\\JPEGImages\\' + self.train_data_ID[i] + '.jpg'
            img = cv2.imread(filename,MODE)
            train_data.append(img)
        offset = self.locateNegSample('dog_train.txt', offset)
        
        #parse negative train_data's ID
        offset = self.readDataID('dog_train.txt', negNum, offset, self.train_data_ID)
        for i in range(0,negNum):
            filename = 'VOCtrainval_06-Nov-2007\\VOCdevkit\\'\
                'VOC2007\\JPEGImages\\' + self.train_data_ID[i+posNum] +'.jpg'
            img = cv2.imread(filename,MODE)
            train_data.append(img)
        
        train_data = np.array(train_data, dtype="object")
        # trainIndex = np.random.choice(trainSize,trainSize,replace = False)
        # self.trainIndex = trainIndex
        # train_data = train_data[trainIndex]
        for i in range(0,trainSize):
            train_data[i] = cv2.resize(train_data[i],(448,448))
    
        # ******************* load test_data ******************#
        test_data = []
        offset = 0
        posNum = negNum = testSize // 2
        
        #parse postive test_data's ID
        offset = self.readDataID('dog_test.txt', posNum, offset, self.test_data_ID)
        
        for i in range(0,posNum):
            filename = 'VOCtest_06-Nov-2007\\VOCdevkit\\'\
                'VOC2007\\JPEGImages\\' + self.test_data_ID[i] + '.jpg'
            img = cv2.imread(filename,MODE)
            test_data.append(img)
        offset = self.locateNegSample('dog_test.txt', offset)
        
        #parse negative test_data's ID
        offset = self.readDataID('dog_test.txt', negNum, offset, self.test_data_ID)
        for i in range(0,negNum):
            filename = 'VOCtest_06-Nov-2007\\VOCdevkit\\'\
                'VOC2007\\JPEGImages\\' + self.test_data_ID[i+posNum] +'.jpg'
            img = cv2.imread(filename,MODE)
            test_data.append(img)
        
        test_data = np.array(test_data, dtype = "object")
        # testIndex = np.random.choice(testSize,testSize,replace = False)
        # self.testIndex = testIndex
        # test_data = test_data[testIndex]
        for i in range(0,testSize):
            test_data[i] = cv2.resize(test_data[i],(448,448))
        
        return train_data, test_data
        
    def readDataID(self, path, size, offset, data_ID):
        f = open(path)
        f.seek(offset)
        line = f.readline()
        i = 0
        while(i < size):
            ID = line[0:6]
            data_ID.append(ID)
            i += 1
            line = f.readline()
        currentOffset = f.tell()
        f.close()
        return currentOffset
            
    def locateNegSample(self, path, offset):
        f = open(path)
        f.seek(offset)
        line = f.readline()
        while(int(line[7:9]) == 1):
            line = f.readline()
        currentOffset = f.tell()
        f.close()
        return currentOffset
        
    def load_label(self,posNum,trainSize):
        trainLabel = np.zeros([trainSize,49,11])
        for i in range(0,posNum):
            size,boundingbox = self.readXML(self.train_data_ID[i])
            # print(boundingbox)
            for coordinate in boundingbox:
                x,y,w,h,X,Y = self.transform(size,coordinate)
                gridID = self.mappingGrid(X,Y)
                data = (x,y,w,h,1)
                j = 0
                for element in data:
                    for k in range(0,2):
                        trainLabel[i,gridID,j + k * 5] = element
                    j += 1
                trainLabel[i,gridID,10] = 1
        # trainLabel = trainLabel[self.trainIndex]
        return trainLabel
        
    def readXML(self,ID):
        boundingbox = []
        filename = 'VOCtrainval_06-Nov-2007\\VOCdevkit\\'\
                'VOC2007\\Annotations\\' + ID + '.xml'
        domTree = parse(filename)
        # 文档根元素
        rootNode = domTree.documentElement
        # print(rootNode.nodeName)
        # 所有物体
        objects = rootNode.getElementsByTagName("object")
        # print("****所有物体信息****")
        for obj in objects:
            # name 元素
            name = obj.getElementsByTagName("name")[0]
            # print(name.nodeName, ":", name.childNodes[0].data)
            if(name.childNodes[0].data == 'dog'):
                size = rootNode.getElementsByTagName("size")[0]
                # width 元素
                width = size.getElementsByTagName("width")[0]
                # height 元素
                height = size.getElementsByTagName("height")[0]
                # bndbox 元素
                bndbox = obj.getElementsByTagName("bndbox")[0]
                
                xmin = bndbox.getElementsByTagName("xmin")[0]
                # print(xmin.nodeName, ":", xmin.childNodes[0].data)
                ymin = bndbox.getElementsByTagName("ymin")[0]
                # print(ymin.nodeName, ":", ymin.childNodes[0].data)
                xmax = bndbox.getElementsByTagName("xmax")[0]
                # print(xmax.nodeName, ":", xmax.childNodes[0].data)
                ymax = bndbox.getElementsByTagName("ymax")[0]
                # print(ymax.nodeName, ":", ymax.childNodes[0].data)
                # print('\n')
                
                coordinate = (int(xmin.childNodes[0].data),
                              int(ymin.childNodes[0].data),
                              int(xmax.childNodes[0].data),
                              int(ymax.childNodes[0].data))
                boundingbox.append(coordinate)
                size = (int(width.childNodes[0].data),
                        int(height.childNodes[0].data))
        return size,boundingbox
                    
    def transform(self,size,coordinate):
        (width,height) = size
        (xmin,ymin,xmax,ymax) = coordinate
        a = 448 / width
        b = 448 / height
        X = round( a*((xmin+xmax)/2 + 1) ) - 1
        x = X % 64 / 64
        Y = round( b*((ymin+ymax)/2 + 1) ) - 1
        y = Y % 64 / 64
        # print(width,height)
        # print(xmin,xmax)
        # print(X,Y)
        
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        return x,y,w,h,X,Y
    
    def mappingGrid(self,X,Y):
        col = X // 64
        # print(col)
        row = Y // 64
        # print(row)
        responsibleGrid = row * 7 + col
        return responsibleGrid
        
                
# head = LoadData()
# # head.readXML('000036')
# train_data,test_data = head.load_image(100,300,400,200)
# trainLabel = head.load_label(100,400)