from common.layers import *
import os
import numpy as np
import collections
import pickle
from common.optimizer import *
from collections import OrderedDict
# from check import numerical_gradient_test
from loss_function import yolo_loss,yolo_return_loss
from process_data import normalize

class Net:
    def __init__(self, batch, lr, momentum):
        self.batch = batch
        self.record_loss = []
        self.record_params = None
        self.params = {}
        self.grads = {}
        self.optimizer = Nesterov(lr, momentum)
        input_dim=(3,448,448)
        conv_param_1 = {'filter_num':1, 'filter_size':1, 'pad':0, 'stride':1}
        conv_param_2 = {'filter_num':64, 'filter_size':7, 'pad':3, 'stride':2}
        conv_param_3 = {'filter_num':128, 'filter_size':2, 'pad':0, 'stride':2}
        conv_param_4 = {'filter_num':256, 'filter_size':2, 'pad':0, 'stride':2}
        hidden_size = 256
        output_size = 11
        pre_channel_num = input_dim[0]
        params_group = [conv_param_1, conv_param_2, conv_param_3, conv_param_4]
        for idx, conv_param in enumerate(params_group):
            n = conv_param['filter_num']
            c_in = pre_channel_num
            h_in = conv_param['filter_size']
            w_in = conv_param['filter_size']
            k = 1 / (c_in * h_in * w_in)
            low = -np.sqrt(k)
            high = np.sqrt(k)
            self.params['W' + str(idx + 1)] = np.random.uniform(low, high, (n, c_in, h_in, w_in))
            self.params['b' + str(idx + 1)] = np.random.uniform(low, high, n)
            pre_channel_num = conv_param['filter_num']
        k = 1 / hidden_size
        low = -np.sqrt(k)
        high = np.sqrt(k)
        self.params['W5'] = np.random.uniform(low, high, (hidden_size, output_size))
        self.params['b5'] = np.random.uniform(low, high, output_size)
        # 生成层===========
        self.layers = collections.OrderedDict()
        self.layers['Conv_1'] = Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad'])
       
        self.layers['Conv_2'] = Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad'])
        # (1, 192, 112, 112)
        self.layers['leaky_Relu_1'] = leaky_Relu()
        # (1, 192, 112, 112)
        self.layers['pooling_1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # (1, 192, 56, 56)
        self.layers['Conv_3'] = Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad'])
        # (1, 128, 56, 56)
        self.layers['leaky_Relu_2'] = leaky_Relu()
        # (1, 128, 56, 56)
        self.layers['pooling_2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # (1, 256, 28, 28)
        self.layers['Conv_4'] = Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad'])
        self.layers['leaky_Relu_3'] = leaky_Relu()
        
        self.layers['pooling_3'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['rp'] = Reshape()

        self.layers['Affine'] = Affine(self.params['W5'], self.params['b5'])
        # self.layers['drop_out'] = Dropout(0.5)
        if(os.path.exists('params.pkl')):
            self.load_params()

    def predict(self,x,train_flg = True):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        self.predict_label = x
        return x

    def loss(self,y,t):
        return yolo_loss(y,t)
    
    def return_loss(self):
        return self.record_loss
    
    def return_params(self):
        return self.record_params
        
    def gradient(self,x,t):
        # forward
        y = self.predict_label

        # backward
        loss_value = self.loss(y,t)
        self.record_loss.append(loss_value)
        print('loss:',loss_value,'\n')
        dout = yolo_return_loss(y,t)
        tmp_layers = list(self.layers.values())
        tmp_layers.reverse()
        
        for layer in tmp_layers:
            dout = layer.backward(dout)

        grads = {}
        layers = list(self.layers.values())
        for i, layer_idx in enumerate((0,1,4,7,11)):   #conv 还有 affine才有参数学习  i是下标01234567   layer_idx是对应的值
            grads['W' + str(i+1)] = layers[layer_idx].dW    #layers 是所有的层 需要在conv   Affine的层数取到  w b
            grads['b' + str(i+1)] = layers[layer_idx].db
        return grads

    def update(self, times, data, label):
        grads_once = self.gradient(data, label)
        for i in range(1,6):
            weight_ID = 'W' + str(i)
            bias_ID = 'b' + str(i)
            self.grads[weight_ID] += grads_once[weight_ID]
            self.grads[bias_ID] += grads_once[bias_ID]
        if times % self.batch == 63:
            for i in range(1,6):
                weight_ID = 'W' + str(i)
                bias_ID = 'b' + str(i)
                self.grads[weight_ID] /= self.batch
                self.grads[bias_ID] /= self.batch
            self.optimizer.update(self.params, self.grads)
            self.reset_grads()
        self.record_params = self.params
        self.save_params()

    def reset_grads(self):
        for i in range(1,6):
            weight_ID = 'W' + str(i)
            bias_ID = 'b' + str(i)
            self.grads[weight_ID] = np.zeros_like(self.params[weight_ID])
            self.grads[bias_ID] = np.zeros_like(self.params[bias_ID])
            
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv_1', 'Conv_2','Conv_3','Conv_4',
                                 'Affine']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]