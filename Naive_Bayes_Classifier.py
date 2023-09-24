# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:29:07 2021

@author: Windows
"""

import pandas as pd
import csv
import numpy as np
from numpy.linalg import inv

TrainingFilename = r'E:\Fall-2021\ML\project1\generated_train.csv'



TestingFilename = r'E:\Fall-2021\ML\project1\generated_test.csv'



import math
import random

def load_csv(filename):
    lines = csv.reader(open(filename, "r", encoding='utf-8-sig'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    dataset = np.asarray(dataset, dtype=np.float32)
    return dataset

def class_sorted_data(dataset):
    classes = np.unique(dataset[:, np.size(dataset, 1) - 1])
    sortedclassdata = []
    for i in range(len(classes)):
        item = classes[i]
        itemindex = np.where(dataset[:, np.size(dataset, 1) - 1] == item)   # index  of rows with label class[i]
        singleclassdataset = dataset[itemindex, 0:np.size(dataset, 1) - 1]  # array  of data for class[i]
        sortedclassdata.append(np.matrix(singleclassdataset))               # matrix of data for class[i]
    return sortedclassdata, classes

def prior_prob(dataset, sortedclassdata):
    priorprobability = []
    for i in range(len(sortedclassdata)):
        priorprobability.append(len(sortedclassdata[i])/len(dataset))
    return priorprobability

def find_mean(sortedclassdata):
    classmeans = []
    for i in range(len(sortedclassdata)):
        classmeans.append(sortedclassdata[i].mean(0))
    return classmeans

def find_covariance(sortedclassdata, classmeans):
    covariance = []
    ndpc = len(sortedclassdata[0])      # total number of data points (rows) per class
    for i in range(len(classmeans)):
        xn = np.transpose(sortedclassdata[i])
        mean_class = np.transpose(classmeans[i])
        tempvariance = sum([(xn[:, x] - mean_class) * np.transpose(xn[:, x] - mean_class) for x in range(int(ndpc))])
        tempvariance = tempvariance / (ndpc - 1)
        covariance.append(tempvariance)
    return covariance

def convert_covariance_to_naive(matrix):
    numofclasses = len(matrix)
    numoffeatures = len(matrix[0])
    for i in range(numofclasses):
        for j in range(numoffeatures):
            for k in range(numoffeatures):
                if j != k:
                    matrix[i][j, k] = 0
    print("Converted covariance to Naive Bayes")
    return matrix

def find_n_class_probability(dataset, classmeans, covariance, priorProb, classes):
    expo = []
    nclassprob = []
    probabilityofclass = []
    datasetDimensions = len(covariance[0])
    testdatasetMatrix = np.matrix(dataset)
    datasetTranspose = np.transpose(testdatasetMatrix[:,0:len(dataset[0])-1])
    for i in range(len(dataset)):
        x = datasetTranspose[:, i]
        for j in range(len(classmeans)):
            determinate = np.linalg.det(covariance[j])
            if determinate == 0:
                addValue = 0.006*np.identity(datasetDimensions)
                covariance[j] = addValue + covariance[j]
                determinate = np.linalg.det(covariance[j])
                #print("Changed Determinate")
            exponent = (-0.5)*np.transpose(x-np.transpose(classmeans[j]))*np.linalg.inv(covariance[j])*(x-np.transpose(classmeans[j]))
            expo.append(exponent)
            nprobabilityofclass = priorProb[j]*(1/((2*math.pi)**(datasetDimensions/2)))*(1/(determinate**0.5))*math.exp(expo[j])
            probabilityofclass.append(nprobabilityofclass)
        arrayprob = np.array(probabilityofclass)
        nclassprob.append(classes[np.argmax(arrayprob)])
        probabilityofclass = []
        expo = []
    return nclassprob

def get_accuracy(nclassprob, dataset):
    Classes = np.transpose([np.asarray(nclassprob, dtype=np.float32)])
    Truth = np.transpose([np.asarray(dataset[:, dataset.shape[1]-1])])
    validate = np.equal(Classes, Truth)
    accuracy = 100 * (np.sum(validate) / dataset.shape[0])
    return accuracy

trainingData = load_csv(TrainingFilename)



testingData = load_csv(TestingFilename)
#testingData = createDataPoint()
sortclassdata, classes = class_sorted_data(trainingData)

priorProb = prior_prob(trainingData, sortclassdata)

meansbyclass = find_mean(sortclassdata)

covariance = find_covariance(sortclassdata, meansbyclass)
covariance = convert_covariance_to_naive(covariance)

nclassprob = find_n_class_probability(trainingData, meansbyclass, covariance, priorProb, classes)
accuracy = get_accuracy(nclassprob, trainingData)
print(f"{accuracy}% Correct on Training Data using Naive Bayes Classifier")

nclassprob = find_n_class_probability(testingData, meansbyclass, covariance, priorProb, classes)

accuracy = get_accuracy(nclassprob, testingData)
print(f"{accuracy}% Correct on Testing Data using Naive Bayes Classifier")

X_train=pd.DataFrame(trainingData[:,:-1])
test_data=pd.DataFrame(testingData[:,:-1])

X_test=test_data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nclassprob=pd.Series(nclassprob)
import matplotlib.pyplot as plt
for class_value in range(2):
    # get row indexes for samples with this class
    row_ix = np.where(nclassprob == class_value)
    colormap = np.array(['r', 'b'])
    # create scatter of these samples
    plt.scatter(X_test[row_ix, 0], X_test[row_ix, 1],c=colormap[class_value],label=f' Class {class_value}')
    plt.legend()
plt.xlabel('X_test')
plt.ylabel('Class Value')
plt.title('Decision Boundary with Naive Bayes Classifier (Generated data)')
#     #plt.show()
# # show the plot
plt.show()

