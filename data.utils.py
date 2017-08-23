# -*- coding:utf-8 -*-
"""
Created on Tue Aug 22 15:50:17 2017

@author: Administrator
"""

import pickle as pickle
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR_batch(filename):
    with open(filename,'rb') as f:
        datadict=pickle.load(f,encoding='latin1')
        #加载文件内容为数据字典形式
        X=datadict['data']
        Y=datadict['labels']
        X.reshape(10000,3,32,32)
        #X为10000*3072（3*32*32）的矩阵，Y为10000的列矩阵
        Y=np.array(Y)
        '''print (X)
        print (Y)'''
        return X,Y
                
def load_CIFAR10(ROOT):
    xs=[]
    ys=[]
    for b in range(1,6):
        f = os.path.join('E:/chuncaicai/cs231n_exercise/cifar_10/cifar-10-batches-py', 'data_batch_%d' % (b, ))
        X,Y=load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
        #将前五个训练集合并
    Xtr=np.concatenate(xs)
    Ytr=np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join('E:/chuncaicai/cs231n_exercise/cifar_10/cifar-10-batches-py', 'test_batch'))
    return Xtr, Ytr, Xte, Yte
                
if __name__ == "__main__":
    imgX, imgY = load_CIFAR_batch('E:/chuncaicai/cs231n_exercise/cifar_10/cifar-10-batches-py/data_batch_1')
    x_train,y_train,x_test,y_test=load_CIFAR10('E:/chuncaicai/cs231n_exercise/cifar_10/cifar-10-batches-py')
    '''print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)'''
    
   
    