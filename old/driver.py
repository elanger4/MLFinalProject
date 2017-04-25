import csv
import numpy as np
import pandas as pd

from math import floor, sqrt
from TimeSeriesNnet import TimeSeriesNnet

items = []
sums = []

with open('product_distribution_training_set.txt', 'rb') as prod_distrib_file:
    prod_distrib_reader = csv.reader(prod_distrib_file, delimiter = '\t')
    for line in prod_distrib_reader:
        temp = []
        for v in range(0, len(line)):
            if v != 0:
                temp.append(float(line[v]))
        items.append(temp)
    prod_distrib_file.close()

with open('key_production_IDs.txt', 'rb') as key_prods_file:
    key_prods_reader = csv.reader(key_prods_file, delimiter = '\t')
    key_products = []
    for line in key_prods_reader:
        key_products.append(line[0])
    key_prods_file.close()


for i in range(0, len(items[0])):
    tempSum = 0
    for j in range(0, len(items)):
        tempSum += items[j][i]
    sums.append(tempSum)


'''
time_series = np.array(sums[:100])
testing = np.array(sums[100:])
'''

time_series = np.array(sums)

training = np.array(sums[:101])
testing = np.array(sums[101:])
print len(time_series)
neural_net = TimeSeriesNnet(hidden_layers = [20, 15, 5], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])

#sys.exit()
#neural_net.fit(time_series, lag = 40, epochs = 1000)
neural_net.fit(training, lag = 50, epochs = 1000)
prediction_vector = neural_net.predict_ahead(n_ahead = 16)

with open('output.txt', 'wb') as output:
    output.write(str(0) + " ")
    for value in prediction_vector:
        output.write(str(value) + " ")
    output.write("\n")
    output.close()


mse = 0
for i, j in zip(testing, neural_net.timeseries[::-1]):
    mse += (i - j) ** 2

'''

for i in range(0, len(prediction_vector)):
    diff = float((testing[i] - prediction_vector[i]) / testing[i] )
    print diff
    mse += (diff * diff)
'''

print "Mean Square Error of total sales: ", sqrt(mse)

import matplotlib.pyplot as plt
testPredictPlot = np.empty(len(time_series))
testPredictPlot.fill(np.nan)
print neural_net.timeseries
testPredictPlot[102:] = neural_net.timeseries[101:118]

plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions', linewidth=1)
plt.plot(range(len(time_series)), time_series, '-g',  label='Original series')
plt.title("Time Series for Prediction for Online Shopping Data")
plt.xlabel("Time (in days)")
plt.ylabel("Sale Quantity")
plt.legend()
plt.draw()

'''
mse_Vector = []

counter = 0
for product in items:

    mse = 0
    diff = 0

#    time_series = np.array(product[:100])
#    testing = np.array(product[100:])
    time_series = np.array(product)

    neural_net = TimeSeriesNnet(hidden_layers = [20, 15, 5], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])

    neural_net.fit(time_series, lag = 50, epochs = 10000)
    prediction_vector = neural_net.predict_ahead(n_ahead = 18)

    with open('output.txt', 'a') as output:
        output.write(str(key_products[counter]) + " ")
        for value in prediction_vector:
            output.write(str(value) + " ")
        output.write("\n")

        output.close()
    counter+=1


'''
plt.savefig('results.png')
plt.show()
