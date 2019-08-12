#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:51:22 2019

@author: fengchi
"""

import numpy as np
import matplotlib.pyplot as plt

class DNN:
    def __ini__(self, nn_shape=(2,4,1)):
        self.W = [] #权重
        self.B = [] #阈值
        self.O = [] #各个神经元的输出
        self.grads = [] #误差与神经元输入的微分（梯度）
        
        self.mean = np.zeros(nn_shape[2])
        self.mean = self.mean.reshape((1, nn_shape[2]))
        
        self.W_shape = []
        self.B_shape = []
        self.O_shape = []
        self.grads_shape = []
        
        self.errs = []
        
        for idx in range(len(nn_shape)-1):
            # 生成0-1之间的浮点数，随机，2x-1以后区间变为[-1,1]，对于原点对称
            self.W.append(2*np.random.random([nn_shape[idx], nn_shape[idx+1]])-1)
            # 将第一个权重元素改为(2,4)类型，表示第一层和第二层之间的神经元网络连接
            self.W[idx] = self.W[idx].reshape([nn_shape[idx], nn_shape[idx+1]])
            self.W_shape.append(self.W[idx].shape)
            
            # B、O、grads一样，都是1行
            self.B.append(2*np.random.random(nn_shape[idx+1])-1)
            self.B[idx] = self.B[idx].reshape(1, nn_shape[idx+1])
            self.B_shape.append(self.B[idx].shape)
            
            self.O.append(np.zeros(nn_shape[idx+1]))
            self.O[idx] = self.O[idx].reshape(1,nn_shape[idx+1])
            self.O_shape.append(self.O[idx].shape)
            
            self.grads.append(np.zeros(nn_shape[idx+1]))
            self.grads[idx] = self.grads[idx].reshape(1,nn_shape[idx+1])
            self.grads_shape.append(self.grads[idx].shape)
        
        # y_hat表示最后一个输出
        self.y_hat = self.O[-1]
        self.y_hat = self.y_hat.reshape(self.O[-1].shape)

        print('建立{}层神经网络'.format(len(nn_shape)))
        print(self.W_shape)
        print(self.B_shape)
        print(self.O_shape)
        print(self.grads_shape)
    
    # x是(1,n)向量
    def sigmoid(self.x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivate(self, x):
        return x * （1 - x)
    
    # 最小平方误差MSE
    def error(self, y, y_hat):
        err = y - y_hat
        return 0.5 * err.dot(err.T)
    
    
    def cross_entropy(self, y, y_hat):
        tmp = np.argwhere(y==1)
        return -np.log(y_hat[0, tmp[0,1])

    def softmax(self, x):
        exp_all = np.exp(x)
        return exp_all / np.sum(exp_all)
    
    
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            