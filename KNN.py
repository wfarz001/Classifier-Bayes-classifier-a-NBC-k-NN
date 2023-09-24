# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 01:27:20 2021

@author: Windows
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def minkowski_distance(a, b, p=1):
    
    # Store the number of dimensions
    dim = len(a)
    
    # Set initial distance to 0
    distance = 0
    
    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
        
    distance = distance**(1/p)
    
    return distance

def knn_predict(X_train, X_test, y_train, y_test, k, p):
    
    # Counter to help with label voting
    from collections import Counter
    
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        
        # Append prediction to output list
        y_hat_test.append(prediction)
        
    return y_hat_test

import csv
filename = r'E:\Fall-2021\ML\project1\generated_train.csv'

mydata = csv.reader(open(filename, "rt"))
mydata = list(mydata)

arr=np.array(mydata) ## Reading data as array
arr=arr.astype(np.float) ## Converting the data type to flaot

arr_data=pd.DataFrame(arr[:,:-1])


arr_cls=pd.Series(arr[:,2])

X_train=arr_data
#y_train=arr_cls
y_train=arr_cls

test_file= r'E:\Fall-2021\ML\project1\generated_test.csv'

test=csv.reader(open(test_file, "rt"))
test=list(test)
test=np.array(test)
test=test.astype(np.float)

test_data=pd.DataFrame(test[:,:-1])

test_cls=pd.Series(test[:,2])

X_test=test_data
y_test=test_cls
y_test=test_cls

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=7, p=1)

y_pred=pd.Series(y_hat_test)

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test, y_pred)

print("Accuracy of  KNN  Model:%.2f%%" %(acc*100.0))


# create scatter plot for samples from each class
for class_value in range(2):
    # get row indexes for samples with this class
    row_ix = np.where(y_pred == class_value)
    colormap = np.array(['r', 'b'])
    # create scatter of these samples
    plt.scatter(X_test[row_ix, 0], X_test[row_ix, 1],c=colormap[class_value],label=f' Class {class_value}')
    plt.legend()
plt.xlabel('X_test')
plt.ylabel('Predicted value')
plt.title('Scatter Plot of KNN Classifier(k=7)')
    #plt.show()
# show the plot
plt.show()


