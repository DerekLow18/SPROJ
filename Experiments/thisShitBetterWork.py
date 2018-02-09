import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import matplotlib.pyplot as plt
import scipy.spatial

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
			#print(dataset[58])
			#print("concat is",toSum)
			runningSum = np.sum(toSum, axis = 0)
		j = j + 1
		for l in range(len(runningSum)):
			if runningSum[l] >1:
				runningSum[l] = 1
		downsampleDataset.append(runningSum)
	downsampleDataset = np.array(downsampleDataset)
	#print(len(downsampleDataset))
	#print(downsampleDataset)
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

print(dataset.shape)
print(dataset)

#intilizalize the weight array
weights = np.random.rand(10,10)
print(weights)



#formula for the prediction of what the next step will look like. Currently, it's at simple thresholding
def activation(activity):
	output = 0
	if activity == 1:
		output = 1
	else: output = 0
	return output

#takes a timeStep and attempts to predict the next time step
def prediction(timeStep):
	global weights, activation
	#matrix multiply the weight matrix with the spiking matrix
	adjustedStep = np.matmul(timeStep, weights)
	#go through all values of the adjusted step matrix, and multiply them by the activation function
	for value in range(len(adjustedStep)):
		adjustedStep[value] = adjustedStep[value]*activation(timeStep[value])
		#print(adjustedStep)
	#return the resulting and final adjusted step
	return adjustedStep

#error calculation between the predicted step and the actual step, euclidean distance
def error(prediction, actual):
	return scipy.spatial.distance.euclidean(prediction, actual)

#main network training function
def trainNetwork(Max_iters = 10):
	for i in range(len(dataset)):
		print("This is the error: " + str(error(prediction(dataset[i]),dataset[i])))

trainNetwork()
