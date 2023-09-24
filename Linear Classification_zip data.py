# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:02:58 2021

@author: Windows
"""

import pandas as pd
import csv
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import OneHotEncoder

filename = r'E:\Fall-2021\ML\project1\zipcode_train.csv'

mydata = csv.reader(open(filename, "rt"))
mydata = list(mydata)

arr=np.array(mydata) ## Reading data as array
arr=arr.astype(np.float) ## Converting the data type to flaot

arr_data=pd.DataFrame(arr[:,:-1])

onehot_encoder = OneHotEncoder(sparse=False)
arr_cls=pd.DataFrame(arr[:,-1:])
arr_cls=onehot_encoder.fit_transform(arr_cls)

X_train=np.matrix(arr_data)
#y_train=arr_cls
y_train=np.matrix(arr_cls)

test_file= r'E:\Fall-2021\ML\project1\zipcode_test.csv'

test=csv.reader(open(test_file, "rt"))
test=list(test)
test=np.array(test)
test=test.astype(np.float)

test_data=pd.DataFrame(test[:,:-1])

test_cls=pd.DataFrame(test[:,-1:])
test_cls=onehot_encoder.fit_transform(test_cls)
X_test=np.matrix(test_data)
y_test=(test_cls)



X_train_transpose=np.transpose(X_train)

k=np.dot(X_train_transpose,X_train)
l=np.linalg.inv(k)
l=np.array(l)

diagonal=l.diagonal()
diagonal=(diagonal+10.0)
diagonal=np.array(diagonal)
row=l.shape[0]
column=l.shape[0]

# import sympy as sp
# h=np.ones((row,column))

for i in  range(row):
    for j in range(column):
        if (i==j):
            l[i,j]=diagonal[i]
    

m=np.dot(X_train_transpose,y_train)

weight=np.dot(l,m)

y_pred_train=np.dot(X_train,weight)
y_pred_train=pd.DataFrame(y_pred_train)
y_pred_train= y_pred_train.eq(y_pred_train.where(y_pred_train!= 0).max(1), axis=0).astype(int)

y_pred=np.dot(X_test,weight)
y_pred=pd.DataFrame(y_pred)
y_preds=y_pred.eq(y_pred.where(y_pred!= 0).max(1), axis=0).astype(int)
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_preds)*100
acc=accuracy_score(y_train,y_pred_train)*100

print("Train Accuracy: %.2f%%" % acc)
print("Testing Accuracy: %.2f%%" % accuracy)
