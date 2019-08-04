#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:57:24 2019

@author: fengchi
"""

import numpy as np
import matplotlib.pyplot as plt

# 读取文本数据，返回数据集和目标值
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(r'testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 该数据集中，添加了一列并初始化为1，便于后续的计算，但是在其他数据集中，一般没有必要添加
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# 运算的核心函数Sigmoid
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

# 核心函数alpha*gradient更新回归系数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn) #将序列转为np的二维数组
    labelMat = np.mat(classLabels).transpose() #将list转为二维数组以后转置
    m,n = np.shape(dataMatrix)
    alpha = 0.001 #设置学习率
    maxCycles = 500 #设置循环次数
    weights = np.ones((n,1)) #初始化每个特征的回归系数为1
    for k in range(maxCycles):
        # 得到每一行的sigmoid值 (两个相乘得到z值)
        h = sigmoid(dataMatrix*weights) # 矩阵相乘 sigmoid(sum(每个特征*每个系数)) 行*列，由于是矩阵相乘，所以在相乘时便求和了
        # 用一直更新的回归系数求预测值，然后与真实值之间求误差，误差越来越小，则系数更新变化也越来越小，最后趋于稳定
        error = labelMat - h
        # 数据集转置*误差 每条样本*该样本的误差值
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 绘图函数
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat) #将mat转换为array list
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)
print(weights)

plotBestFit(weights.getA()) #getA()为将矩阵类型转换为Array数组