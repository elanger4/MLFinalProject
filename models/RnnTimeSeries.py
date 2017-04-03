import matplotlib.pyplot as plt
from pandas import DataFrame, read_table
from keras.models import Sequential
from keras.layers import LSTM, Dense 
from sklearn.preprocessing import MinMaxScaler

data = read_table('product_distribution_training_set.txt', sep='\t', header=None).convert_objects(convert_numeric=True)

def mean_squared_error(y_pred, y_true):
    return np.average((y_pred - y_true) ** 2)

for column in data.columns:
    if data[column].isnull().values.any():
        data[column].fillna(int(data[column].mean()), inplace=True)

data_avgs = [ sum(data[column]) for column in data.drop(data.columns[0], axis=1) ]

'''
fig = plt.figure()
plt.plot(data_avgs)
fig.savefig('data_means.pdf')
plt.show()
'''
