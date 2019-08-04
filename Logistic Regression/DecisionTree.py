#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:37:51 2019

@author: fengchi
"""

from math import log
import operator
import pickle

def calc_shannon_ent(dataset):
    rows_num = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        curr_label = feat_vec[-1]
        if curr_label not in label_counts.keys():
            label_counts[curr_label] = 0
        label_counts[curr_label] += 1
    shannon_ent = 0
    for key in label_counts:
        prob = float(label_counts[key]) / rows_num #选择这一类别的概率
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def create_dataset():
    # 构建数据集
    dataset = [[0, 0, 0, 0, 'no'], 
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况'] #特征标签
    return dataset, labels

def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            rest_feat_vec = feat_vec[:axis] #drop掉被选中的特征
            rest_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(rest_feat_vec)
    return ret_dataset

def choose_best_feat_to_split(dataset):
    num_feat = len(dataset[0]) - 1 #特征数量
    base_ent = calc_shannon_ent(dataset) #计算香农熵
    best_info_gain = 0 #初始化信息增益
    best_feat = -1
    for idx in range(num_feat):
        feat_list = [data[idx] for data in dataset]
        unique_val = set(feat_list) #创建set集合，保证值不重复！！！
        new_ent = 0.0
        for val in unique_val: #计算去掉每个特征以后的信息增益
            sub_dataset = split_dataset(dataset, idx, val)
            prob = len(sub_dataset) / float(len(dataset))
            new_ent += prob * calc_shannon_ent(sub_dataset) #根据公式计算经验条件熵
        info_gain = base_ent - new_ent
        if(info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feat = idx
        return best_feat

# 返回出现次数最多的类别
def max_class_cnt(class_list):
    class_cnt = {}
    for vote in class_list:
        if vote not in class_cnt.keys():
            class_cnt[vote] = 0
    # 根据字典的value排序
    sorted_class_cnt = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_cnt[0][0]

# 创建决策树（递归）
def create_tree(dataset, labels, feat_labels):
    class_list = [data[-1] for data in dataset]
    if class_list.count(class_list[0]) == len(class_list): #如果类别完全相同的情况
        return class_list[0]
    if len(dataset[0]) == 1 or len(labels) == 0: #遍历完所有特征时
        return max_class_cnt(class_list)
    best_feat = choose_best_feat_to_split(dataset)
    best_feat_label = labels[best_feat]
    feat_labels.append(best_feat_label)
    ret_tree = {best_feat_label:{}}
    del(labels[best_feat])
    feat_val = [data[best_feat] for data in dataset]
    unique_val = set(feat_val)
    for value in unique_val:
        sub_labels = labels[:]
        ret_tree[best_feat_label][value] = create_tree(split_dataset(dataset,\
                best_feat, value), sub_labels, feat_labels)
    return ret_tree
    
## 获得树的深度（递归）
def get_tree_depth(tree):
    max_depth = 0
    first_node = next(iter(tree))
    second_dict = tree[first_node]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

## 获取树的叶子树（递归）
def get_tree_leafs(tree):
    num_leaf = 0
    first_node = next(iter(tree))
    second_dict = tree[first_node]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leaf += get_tree_leafs(second_dict[key])
        else:
            num_leaf += 1 #判断不是叶子节点，这时候+1
    return num_leaf

## 使用训练好的决策树进行分类(递归)
def classify(input_tree, feat_labels, test_vec):
    first_node = next(iter(input_tree))
    second_dict = input_tree[first_node]
    feat_index = feat_labels.index(first_node)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label

## 保存模型
def save_model(input_tree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(input_tree, fw)
        
## 加载模型
def load_model(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    dataset, labels = create_dataset()
    feat_labels = []
    my_tree = create_tree(dataset, labels, feat_labels)
    save_model(my_tree, 'model.pkl')
    my_tree = load_model('model.pkl')
    print(my_tree)
    test_vec = [0,1,0,0]
    result = classify(my_tree, feat_labels, test_vec)
    if result == 'yes':
        print('放贷')
    else:
        print('不放贷')