# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:02:29 2021

@author: Windows
"""

import pandas as pd
import csv
import numpy as np
from numpy.linalg import inv

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
filename = r'E:\Fall-2021\ML\project1\regression_train.csv'




test_file= r'E:\Fall-2021\ML\project1\regression_tst.csv'

mydata = csv.reader(open(filename, "rt"))
mydata = list(mydata)

arr=np.array(mydata) ## Reading data as array
arr=arr.astype(np.float) ## Converting the data type to flaot

arr_data=pd.DataFrame(arr[:,:-7])


arr_cls=pd.DataFrame(arr[:,-7:])

X_train=np.matrix(arr_data)
#y_train=arr_cls
y_train=np.matrix(arr_cls)




test=csv.reader(open(test_file, "rt"))
test=list(test)
test=np.array(test)
test=test.astype(np.float)

test_data=pd.DataFrame(test[:,:-7])

test_cls=pd.DataFrame(test[:,-7:])

X_test=np.matrix(test_data)
y_test=np.matrix(test_cls)
#y_test=test_cls




X_train_transpose=np.transpose(X_train)

k=np.dot(X_train_transpose,X_train)
l=np.linalg.inv(k)

m=np.dot(X_train_transpose,y_train)

weight=np.dot(l,m)

y_pred_tr=np.dot(X_train,weight)
y_pred=np.dot(X_test,weight)

MSE_train=np.square(np.subtract(y_train,y_pred_tr)).mean()

MSE = np.square(np.subtract(y_test,y_pred)).mean()

print("Mean Squre Error of Training Data",MSE_train)
print("Mean Squre Error of Testing Data",MSE)