from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from pandas import DataFrame
from math import sqrt
from sklearn.metrics import mean_squared_error
import os


#data.plot()
#pyplot.show()
#This helps determine our second paramenter. Since the graph is vaguely quadratic we need a value of 1


# autocorrelation_plot(data)
# pyplot.show()
# This helps us determine our first parameter of our Arima function which I found we should set to about 10

#arima = ARIMA(data,order=(5,2,1))
#model_fit = arima.fit(disp=0)
#print(model_fit.summary())



def daysToDates(val):
	return datetime.strptime("190"+str(val), '%Y-%m')

def findRMSE():
	totalTestSet = []
	totalPredictions = []
	totalError = []
	for filename in os.listdir("data/"):
		data = read_csv("data/" + str(filename), header = 0, parse_dates=[0], index_col = 0, squeeze = True, date_parser = daysToDates)
		sales = data.values
		trainingSet = sales[:100]
		testSet = sales[100:]
		updatingTrainingSet = []
		print "----"
		for i in range(0,100):
			updatingTrainingSet.append(sales[i])
		predictions = []
		for i in range(0,16):
			arimaModel = ARIMA(updatingTrainingSet,order=(7,1,0))
			fittedModel = arimaModel.fit(disp=0)
			prediction = fittedModel.forecast()[0]
			predictions.append(prediction)
			updatingTrainingSet.append(testSet[i])
			totalTestSet.append(testSet)
			totalPredictions.append(predictions)

		squaredError =mean_squared_error(testSet, predictions)
		totalError.append(squaredError)

	err = 0
	for i in range(len(totalError)):
		err = err + totalError[i]
	error = sqrt(err)
	print "The RMSE is " + str(error)

def makePredictions():

	totalPredictions = []
	for i in range(118,146):
		totalPredictions.append(0.0)
	print len(totalPredictions)
	output = open("dataDocument", 'w')	
	for filename in os.listdir("data/"):
		output.write("\n\n Key Product: " + str(filename) + "\n")
		data = read_csv("data/" + str(filename), header = 0, parse_dates=[0], index_col = 0, squeeze = True, date_parser = daysToDates)
		trainingSet = data.values
		updatingTrainingSet = []
		for i in range(0,116):
			updatingTrainingSet.append(trainingSet[i])
		predictions = []
		for i in range(118,146):
			arimaModel = ARIMA(updatingTrainingSet,order=(7,1,0))
			fittedModel = arimaModel.fit(disp=0)
			prediction = fittedModel.forecast()[0]
			if prediction[0] < 0:
				prediction[0] = 0
			totalPredictions[i-118] = totalPredictions[i-118] + prediction[0]
			predictions.insert(len(predictions),prediction[0])
			updatingTrainingSet.append(prediction)
			#print( str(prediction) + " , " +str(testSet[i]))
		for i in range(len(predictions)):		
			output.write(str(round(predictions[i],4)) + " , " )

	output.write("\n\n\n\n")
	for i in range(len(totalPredictions)):
		output.write(str(round(totalPredictions[i],4)) + " , ")
	



makePredictions()


