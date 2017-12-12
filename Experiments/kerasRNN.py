#keras implementation of RNN for predicting the next time step in a time series

#this is essentially a time series analysis, where the spiking ID is related to the time step
#of the simulated data

# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

dataframe = pandas.read_csv('./Spike Results/1idTimes.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')
#scaler = MinMaxScaler(feature_range = (0,1))
#dataset = scaler.fit_transform(dataset)
print(dataset)
dataset = np.transpose(dataset)
print(len(dataset))

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

batchSize = 9
trainX, trainY = train[0:len(train)-1], train[1:len(train)]
print(trainX[58])
print(trainY[57])
testX, testY = test[0:len(test)-1], test[1:len(test)]
#print(trainX[0])
print(trainX.shape)
print(testX.shape)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape)
print(trainY.shape)

#Create the network here. Each hidden layer has 10 LSTM blocks, with 1 input, and a single output layer
model = keras.models.Sequential()
model.add(keras.layers.LSTM(10,input_shape=(1, batchSize),return_sequences=True))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(9))
model.add(keras.layers.ThresholdedReLU(theta = 0.5))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs = 10, batch_size = 1, verbose = 2)

#run the model here
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
model.summary()
print("Inputs: {}".format(model.input_shape))
print("Outputs: {}".format(model.output_shape))
print("Actual input: {}".format(trainX.shape))
print("Actual output: {}".format(trainPredict.shape))
print(repr(testY[57]))
print(repr(testPredict[57]))
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

