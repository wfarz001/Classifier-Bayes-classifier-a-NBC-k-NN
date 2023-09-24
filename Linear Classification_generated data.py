# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 22:48:15 2021

@author: Windows
"""


import pandas as pd
import csv
import numpy as np
from numpy.linalg import inv

filename = r'E:\Fall-2021\ML\project1\generated_train.csv'

mydata = csv.reader(open(filename, "rt"))
mydata = list(mydata)

arr=np.array(mydata) ## Reading data as array
arr=arr.astype(np.float) ## Converting the data type to flaot

arr_data=pd.DataFrame(arr[:,:-1])


arr_cls=pd.DataFrame(arr[:,2])

X_train=pd.DataFrame(arr_data)
bias=np.ones(400,dtype=float)
bias=list(bias)
X_train['bias']=bias
first_column = X_train.pop('bias')
X_train.insert(0, 'Bias', first_column)


X_train=np.matrix(X_train)

# #y_train=arr_cls
y_train=np.matrix(arr_cls)

test_file= r'E:\Fall-2021\ML\project1\generated_test.csv'

test=csv.reader(open(test_file, "rt"))
test=list(test)
test=np.array(test)
test=test.astype(np.float)

test_data=pd.DataFrame(test[:,:-1])

test_cls=pd.DataFrame(test[:,2])

X_test=pd.DataFrame(test_data)
bias=np.ones(400,dtype=float)
bias=list(bias)
X_test['bias']=bias
first_column = X_test.pop('bias')
X_test.insert(0, 'Bias', first_column)

X_test=np.matrix(X_test)
y_test=np.matrix(test_cls)



X_train_transpose=np.transpose(X_train)

k=np.dot(X_train_transpose,X_train)
l=np.linalg.inv(k)

m=np.dot(X_train_transpose,y_train)

weight=np.dot(l,m)

y_pred_train=np.dot(X_train,weight)
y_pred_train=y_pred_train.astype(int)
y_pred=np.dot(X_test,weight)
y_preds=y_pred.astype(int)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_preds)*100
acc_train=accuracy_score(y_train,y_pred_train)*100
print(" Testing Accuracy: %.2f%%" % accuracy)
print(" Training Accuracy: %.2f%%" % acc_train)