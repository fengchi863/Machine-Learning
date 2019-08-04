#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 09:22:28 2019

@author: fengchi
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    with open('lenses.txt') as fr:
        lenses = [line.strip().split('\t') for line in fr.readlines()]
    lenses_target = []
    for example in lenses:
        lenses_target.append(example[-1])
    
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lenses_labels:
        for each in lenses:
            lenses_list.append(each[lenses_labels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    print(lenses_dict)
    pd_lenses = pd.DataFrame(lenses_dict)
    print(pd_lenses)
    le = LabelEncoder()
    for col in pd_lenses.columns:
        pd_lenses[col] = le.fit_transform(pd_lenses[col]) #为每一列序列化
    print(pd_lenses)
    
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(pd_lenses, lenses_target) 
    
    print(clf.predict([[1,1,1,0]]))c