#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:51:22 2019

@author: fengchi
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

class DNN:
    def __init__(self, nn_shape=(2,4,1)):
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
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivate(self, x):
        return x * (1 - x)
    
    # 最小平方误差MSE
    def error(self, y, y_hat):
        err = y - y_hat
        return 0.5 * err.dot(err.T)
    
    def cross_entropy(self, y, y_hat):
        tmp = np.argwhere(y==1)
        return -np.log(y_hat[0, tmp[0,1]])

    def softmax(self, x):
        exp_all = np.exp(x)
        return exp_all / np.sum(exp_all)
    
    def update_output(self, x, x_istest=False):
        if x_istest == True:
            x = (x - self.mean) / self.var
        for idx in range(len(self.O)):
            if idx == 0:
                self.O[idx] = self.sigmoid(
                        x.dot(self.W[idx]) + self.B[idx])
            elif idx == len(self.O) - 1:
                self.O[idx] = self.softmax(
                        self.O[idx - 1].dot(self.W[idx]) + self.B[idx])
            else:
                self.O[idx] = self.sigmoid(
                        self.O[idx - 1].dot(self.W[idx]) + self.B[idx])
            
            self.O[idx] = self.O[idx].reshape(self.O_shape[idx])
            
        self.y_hat = self.O[-1]
        self.y_hat = self.y_hat.reshape(self.O[-1].shape)
        return self.y_hat
    
    def update_grads(self, y):
        for idx in range(len(self.grads)-1, -1, -1):
            if idx == len(self.grads) - 1:
                # 该代码用来计算使用均方误差和sigmoid函数的二分类问题
#                self.grads[index] = self.sigmoid_derivate(
#                        self.O[index]) * (y - self.O[index])
                tmp = np.argwhere(y==1)
                for idx_g in range(self.grads[idx].shape[1]):
                    if idx_g == tmp[0,1]:
                        self.grads[idx][0, idx_g] = 1 - self.O[idx][0, idx_g]
                    else:
                        self.grads[idx][0, idx_g] = -self.O[idx][0, idx_g]
            else:
                self.grads[idx] = self.sigmoid_derivate(
                        self.O[idx]) * self.W[idx+1].dot(self.grads[idx+1].T).T
            self.grads[idx] = self.grads[idx].reshape(self.grads_shape[idx])
    
    def update_WB(self, x, learning_rate):
        for idx in range(len(self.W)):
            if idx == 0:
                self.W[idx] += learning_rate * x.T.dot(self.grads[idx])
                self.B[idx] -= learning_rate * self.grads[idx]
            else:
                self.W[idx] += learning_rate * self.O[idx-1].T.dot(self.grads[idx])
                self.B[idx] -= learning_rate * self.grads[idx]
            self.B[idx] = self.B[idx].reshape(self.B_shape[idx])
    
    def preprocess(self, X, method='centring'):
        self.mean = np.mean(X, axis=0)
        self.var = X.var()
        X = (X - self.mean) / self.var
        if method == 'centring':
            return X

    def fit(self, X, Y, Preprocess=True, method='centring', thre=0.03, learning_rate=0.001, max_iter=1000):
        if Preprocess == True:
            X = self.preprocess(X,method=method)
        err = np.inf
        count = 0
        while err > thre:
            err = 0
            for idx in range(X.shape[0]):
                x = X[idx,:].reshape((1,-1))
                y = Y[idx,:].reshape((1,-1))
                
                self.update_output(x)
                x = X[idx,:].reshape((1,-1))
                self.update_grads(y)
                self.update_WB(x, learning_rate=learning_rate)
                err += self.cross_entropy(y, self.y_hat)
            err /= idx + 1
            self.errs.append(err)
            count += 1
            if count > max_iter:
                print("超过最大迭代次数{}".format(max_iter))
                break
            
            print(count)
            print(err)
            
    def one_hot_label(self, Y):
        category = list(set(Y[:,0]))
        Y_ = np.zeros([Y.shape[0], len(category)])
        
        for idx in range(Y.shape[0]):
            Y_[idx, Y[idx, 0]] = 1
        
        return Y_
    
if __name__ == '__main__':
    
    digits = load_digits()
    X = digits.data
    Y = digits.target
    X = X.reshape(X.shape)
    Y = Y.reshape(Y.shape[0], 1)
    bp = DNN([64,128,64,10])
    Y = bp.one_hot_label(Y)
    
    train_data = X[:1000, :]
    train_label = Y[:1000, :]
    
    test_data = X[1000:-1, :]
    test_label = Y[1000:-1, :]
    
    bp.fit(train_data, train_label, Preprocess=True, thre=0.01,learning_rate=0.005,
           max_iter=1000)
    count = 0
    for idx in range(test_data.shape[0]):
        x = test_data[idx].reshape(1,64)
        pre = bp.update_output(x, x_istest=True)
        y = test_label[idx].reshape(1,10)
        a = np.where(pre==np.max(pre)) #获得最大的数字的坐标
        b = np.where(y==np.max(y))
        if a[1][0] == b[1][0]:
            count += 1
        
    print('准确率:{}'.format(count/test_label.shape[0]))
    plt.plot(bp.errs)
    plt.show()
    
                    
            
            
            
            
            
            
            
            
            
            
            
            
            
            