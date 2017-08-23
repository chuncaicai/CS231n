# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:31:44 2017

@author: Administrator
"""

import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass
    
    def train(self,X,y):
        """X是N*D矩阵，y是N*1矩阵"""
        #X是图片，y是标签
        self.Xtr=X
        self.ytr=y
        
    def predict(self,X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test,dtype = self.ytr.dtype)
        
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i:]),axis = 1)
            min_index = np.argmin(distances)
            Ypred[i]=self.ytr[min_index]
        return Ypred

