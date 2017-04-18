import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot
from pandas import DataFrame
from math import sqrt
from sklearn.metrics import mean_squared_error
import os
import re

def quad_reg(order):
    totalTrainingSet = []
    keys = []

    #get the training data from "product_distribution_training_set.txt"

    trainingSet  = open("../data/product_distribution_training_set.txt")
    line = trainingSet.readline()
    while(line):
        line = line.rstrip()
        splitList = re.split(r'\t+', line.rstrip('\t'))
        keys.append(splitList[0])
        del splitList[0]
        totalTrainingSet.append(splitList)
        line = trainingSet.readline()

    #totalTrainingSet now contains a list of lists of values for each product on each day

    totalSquaredError = []
    for data in totalTrainingSet:
        data = list(map(int, data)) #converts to ints
        trainSet = data[:100] #separate into training and test sets
        testSet = data[100:]
        updateTrainSet = []
        for i in range(0,100):
            updateTrainSet.append(data[i])
            
        x = np.linspace(0,100,100)
        X = x[:, np.newaxis]
        newx = np.linspace(100,118,18)
        NEWX = newx[:, np.newaxis]
        model = make_pipeline(PolynomialFeatures(order), Ridge())
        model.fit(X, trainSet)
        predictions = model.predict(NEWX)
        squaredError = mean_squared_error(testSet, predictions)

        totalSquaredError.append(squaredError)
        
    error = 0
    for i in range(len(totalSquaredError)):
        error += totalSquaredError[i]
        
    error = sqrt(error)
    print("The RMSE for quadratic regression with order = " + str(order) + " is: " + str(error))
    ''' 
    The RMSE for quadratic regression with order = 2 is: 267.741120655
    The RMSE for quadratic regression with order = 3 is: 294.904854136
    The RMSE for quadratic regression with order = 4 is: 250.177665511
    The RMSE for quadratic regression with order = 5 is: 446.388899503
    The RMSE for quadratic regression with order = 6 is: 812.418975318
    
    Order = 4 was chosen

    '''
    
    #Now, predict for days 119-146
    count = 0
    output = open("DataDoc", 'w')
    for data in totalTrainingSet:
        data = list(map(int, data)) #converts to ints  
        output.write("\n\n Key Product: " + str(keys[count]) + "\n")
        x = np.linspace(0,118,118)
        X = x[:, np.newaxis]
        newx = np.linspace(119,146,27)
        NEWX = newx[:, np.newaxis]
        model = make_pipeline(PolynomialFeatures(order), Ridge())
        model.fit(X, data)
        predictions = model.predict(NEWX)
        for i in range(len(predictions)):
            if(predictions[i] < 0):                                                                                                                                          
                predictions[i] = 0   
        for prediction in predictions:
            output.write(str(round(prediction,4)) + " ,")

        count += 1


    '''

    x = np.linspace(0,118,118)

    #plt.scatter(x, data, color='navy', s=30, marker='o', label="training points")
    print(len(data))

    X = x[:, np.newaxis]

    newx = np.linspace(0,135,135);
    NEWX = newx[:, np.newaxis]

    model = make_pipeline(PolynomialFeatures(order), Ridge())
    model.fit(X, data)
    dataLine = model.predict(NEWX)

    #normalizes the data so that there are no negative values
    for i in range(len(dataLine)):
        if(dataLine[i] < 0):
            dataLine[i] = 0
    print(dataLine)
    plt.plot(newx, dataLine, color="black", linewidth=2, label="degree %d" % order)
    plt.show()
'''    
    

quad_reg(5)
