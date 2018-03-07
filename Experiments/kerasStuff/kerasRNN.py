#keras implementation of RNN for predicting the next time step in a time series

#this is essentially a time series analysis, where the spiking ID is related to the time step
#of the simulated data

import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def downsample(dataset, binSize):
	downsampleDataset = []
	print(len(dataset))
	j=0
	for i in range(int(len(dataset)/10)):
		runningSum = np.zeros(len(dataset[0]))
		for k in range(j*10, (j+1)*10):
			toSum = np.concatenate(([runningSum],[dataset[k]]))
			runningSum = np.sum(toSum, axis = 0)
		j = j + 1
		for l in range(len(runningSum)):
			if runningSum[l] >1:
				runningSum[l] = 1
		downsampleDataset.append(runningSum)
	downsampleDataset = np.array(downsampleDataset)
	print(len(downsampleDataset))
	print(downsampleDataset)
	np.savetxt('downSampledData.csv', downsampleDataset.transpose(), delimiter=",")
	return downsampleDataset

dataset = np.genfromtxt('./Spike Results/1idTimes.csv', delimiter = ',')
print(dataset.shape)
#scaler = MinMaxScaler(feature_range = (0,1))
#dataset = scaler.fit_transform(dataset)
dataset = np.transpose(dataset)
print(dataset.shape)
#now, each array in dataset is representative of a single timestep, where each value is whether or not the neuron is spiking at that particular time

#bin the data set so that every set of 10 steps is reshaped to be one step, and multiple spikes during that time only counted as 1
dataset = downsample(dataset,10)
testset = np.genfromtxt('./Spike Results/2idTimes.csv', delimiter = ',')
print(testset.shape)
#scaler = MinMaxScaler(feature_range = (0,1))
#dataset = scaler.fit_transform(dataset)
testset = np.transpose(testset)
print(testset.shape)
#now, each array in dataset is representative of a single timestep, where each value is whether or not the neuron is spiking at that particular time

#bin the data set so that every set of 10 steps is reshaped to be one step, and multiple spikes during that time only counted as 1
testset = downsample(testset,10)

#length of data set is 1000, so it takes 2/3 of that data set for training
#train_size = int(len(dataset) * 0.67)
#test_size = len(dataset) - train_size
#the first 2/3 of the dataset is used for training, the last third used for testing
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train = dataset
test = testset

popSize = 10
trainX, trainY = train[0:len(train)-1], train[1:len(train)]
#now I have designated at the inputs are all arrays from the first step to the second to last step,
#and the corresponding outputs are all arrays from second step to the last step
testX, testY = test[0:len(test)-1], test[1:len(test)]
#the second value, designating that number of neurons in each timestep, should be the same, and indeed they are
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print("train X shape is:",trainX.shape)
print("train y shape is:",trainY.shape)

#Create the network here. Each hidden layer has 10 LSTM blocks, with 1 input, and a single output layer
model = keras.models.Sequential()
model.add(keras.layers.LSTM(10,input_shape=(1, popSize),return_sequences=True))
model.add(keras.layers.Flatten())
#define the output space using Dense
model.add(keras.layers.Dense(10,activation= 'linear'))
#model.add(keras.layers.ThresholdedReLU(theta = 0.5))
model.compile(loss='binary_crossentropy', optimizer='RMSprop')
model.fit(trainX, trainY, epochs = 10, batch_size = 1, verbose = 2)

#run the model here
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
model.summary()
print("Inputs: {}".format(model.input_shape))
print("Outputs: {}".format(model.output_shape))
print("Actual input: {}".format(trainX.shape))
print("Actual output: {}".format(trainPredict.shape))
print("testPredict shape is",testPredict.shape)
predictedOutput = testPredict.transpose()
predictedMax, predictedMin = predictedOutput.max(), predictedOutput.min()
prediction = (predictedOutput - predictedMin)/(predictedMax - predictedMin)
#print(prediction)
#simple thresholding, I won't pretend to think this is anything meaningful, but its a start, and covers the basics
for i in range(len(prediction)):
	for j in range(len(prediction[i])):
		if prediction[i][j] >= 0.5:
			prediction[i][j] = 1
		elif prediction [i][j] <0.5:
			prediction[i][j] = 0
np.savetxt('kerasPrediction.csv', predictedOutput, delimiter=",")
np.savetxt('kerasThresholded.csv', prediction, delimiter=",")

# calculate root mean squared error for testing pre and post thresholding
trainScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Non-thresholded: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, prediction.transpose()))
print('Thresholded: %.2f RMSE' % (testScore))
#plt.vlines(testY.transpose())
#plt.show()
#plt.eventplot(prediction.transpose())
#plt.show()

