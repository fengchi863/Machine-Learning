# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:06:46 2019

@author: fengchi863
"""

import tokenFile
import numpy as np

def softmax(x):
    x = np.array(x)
    max_x = np.max(x) #防止程序溢出，造成error，可以证明减与不减两者相等
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))

class RNN:
    def __init__(self, data_dim, hidden_dim=100, bptt_back=4):
        # data_dim: 词向量维度，即词典长度; hidden_dim: 隐单元维度; bptt_back: 反向传播回传时间长度
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.bptt_back = bptt_back
        
        # 初始化权重向量 U， W， V; U为输入权重; W为递归权重; V为输出权重
        self.U = np.random.uniform(-np.sqrt(1.0/self.data_dim), np.sqrt(1.0/self.data_dim), 
                                   (self.hidden_dim, self.data_dim))
        self.W = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), 
                                   (self.hidden_dim, self.hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), 
                                   (self.data_dim, self.hidden_dim))
    
    def forward(self, x):
        T = len(x)
        
        # 初始化状态向量, s包含额外的初始状态 s[-1]
        s = np.zeros((T+1, self.hidden_dim))
        o = np.zeros((T, self.data_dim))
        
        for t in range(T): #xrange为返回一个迭代器，range直接返回list数组
            s[t] = np.tanh(self.U[:,x[t]]) + self.W.dot(s[t-1])
            o[t] = softmax(self.V.dot(s[t]))
        
        return [o, s]
    
    def predict(self, x):
        o, s = self.forward(x)
        pre_y = np.argmax(o, axis=1)
        return pre_y

    def loss(self, x, y):
        cost = 0
        for i in range(len(y)):
            o, s = self.forward(x[i])
            
        