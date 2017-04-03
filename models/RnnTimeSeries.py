import matplotlib.pyplot as plt
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

# Replace any values that are unavailable / clean the data
for column in data.columns:
    if data[column].isnull().values.any():
        data[column].fillna(int(data[column].mean()), inplace=True)

# Compute the average for each
data_avgs = [ sum(data[column]) for column in data.drop(data.columns[0], axis=1) ]

# In order to avoid exploding gradients, scale the data between 0 -> 1
#data = DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(data))

training = data_avgs[:101]
validation = data_avgs[101:]

train_in = np.array([ [i] for i in training[:-1] ])
train_out  = np.array([ [i] for i in training[1:] ])
print train_in
print train_out

train_in = train_in.reshape((1,100,1))
train_out = train_out.reshape((1,100))

val_in  = [ [i] for i in validation[1:] ]
val_out = [ [i] for i in validation[:-1] ]

# Split dataset into trainging and validation data
training = data_avgs[:101]
validation = data_avgs[101:]

model = Sequential()
model.add(LSTM(100, input_shape=(1,100)))
model.add(Dense(100))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_in, train_out, batch_size=1, verbose=2)

'''
trainPredict = model.predict(training)
testPredict = model.predict(validation)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

fig = plt.figure()
plt.plot(data_avgs)
fig.savefig('data_means.pdf')
plt.show()
'''
