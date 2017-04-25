import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense 
from pandas import DataFrame, read_table
from sklearn.preprocessing import MinMaxScaler

data_file = os.getcwd() + '/' + sys.argv[1]
# Reads in data and converts strings to floats
data = read_table(data_file, sep='\t', header=None).convert_objects(convert_numeric=True)

# Function to calculate our prediction error
def mean_squared_error(y_pred, y_true):
    return np.average((y_pred - y_true) ** 2)

# convert an array of values into a dataset matrix
'''
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)
'''

# Replace any values that are unavailable / clean the data
for column in data.columns:
    if data[column].isnull().values.any():
        data[column].fillna(int(data[column].mean()), inplace=True)

# Compute the average for each
data_avgs = np.array([ sum(data[column]) for column in data.drop(data.columns[0], axis=1) ])
data_avgs = np.reshape(data_avgs, (len(data_avgs), 1))


scaler = MinMaxScaler(feature_range=(0,1))

#data_avgs = scaler.fit_transform(data_avgs.reshape(1, -1))
data_avgs = scaler.fit_transform(data_avgs)

training = data_avgs[:101]
validation = data_avgs[101:]
# In order to avoid exploding gradients, scale the data between 0 -> 1
#data_avgs = DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(data_avgs))


# Split dataset into trainging and validation data
#print data_avgs

trainX = np.empty([len(training) - 1, 1], dtype=float)
testX = np.empty([len(validation) - 1, 1], dtype=float)

for i, val in enumerate(training[:-1]):
    trainX[i] = np.array(np.array(val))

for i, val in enumerate(validation[:-1]):
    testX[i] = (np.array(np.array(val)))

#trainX = np.matrix([np.array(i) for i in training[:-1]])


trainY = np.array([i[0] for i in training[1:]])
testY = np.array([i[0] for i in validation[1:]])

back = 1
'''
trainX, trainY = create_dataset(training, back)
testX, testY = create_dataset(validation, back)

testX  = np.reshape(testX,  (testX.shape[0],  1, testX.shape[1]))
'''
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX =  np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

'''
trainPredictPlot = np.empty_like(data_avgs)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[back:len(trainPredict)+back, :] = trainPredict

testPredictPlot = np.empty_like(data_avgs)
testPredictPlot[:, :] = np.nan
testPredictPlot[back:len(trainPredict)+back*2+1: len(data_avgs)-1, :] = testPredict

plt.plot(scaler.inverse_transform(data_avgs))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
'''
testPredictPlot = np.empty_like(data_avgs)
testPredictPlot[:, :] = np.nan
testPredictPlot[102:] = testPredict

plt.plot(scaler.inverse_transform(data_avgs))
plt.plot(trainPredict)
plt.plot(testPredictPlot)
plt.savefig('rnn_predict.png')
plt.show()
