import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import matplotlib.pyplot as plt
import scipy.spatial
from sympy import symbols, diff	

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

#error calculation between the predicted step and the actual step, euclidean distance
def error(prediction, actual):
	return scipy.spatial.distance.euclidean(prediction, actual)

def pdEuclideanDistance(predicted,actual):#calculate value for partial deriv of euclidean distance w.r.t. predicted
	return (predicted-actual)/(np.sqrt((predicted-actual)**2))

#formula for the prediction of what the next step will look like. Currently, it's at sigmoind function
def activation(activity):
	return 1 / (1 + math.exp(-activity))

def pdSigmoid(x):
	return x*(1-x)

#takes a timeStep and attempts to predict the next time step
def prediction(timeStep):
	global weights, activation
	#matrix multiply the weight matrix with the spiking matrix
	adjustedStep = np.matmul(timeStep, weights)
	#go through all values of the adjusted step matrix, and multiply them by the activation function
	for value in range(len(adjustedStep)):
		adjustedStep[value] = activation(adjustedStep[value])
		#print(adjustedStep)
	#return the resulting and final adjusted step
	return adjustedStep

'''
change the weight between one source neuron and the target neuron

'''
def weightChange(predicted,actual,activity,priorStep):
	'''x, y= symbols('x y', real=True)
	f = ((x-y)**2)**(1/2)
	deriv = diff(f, x)
	print(deriv)'''
	i = pdEuclideanDistance(predicted,actual) #partial derivative of euclidean distance with respect to the prediction
	j = pdSigmoid(activation(activity)) #partial derivative of activation function with respect to the activity
	totalChange = i*j*priorStep
	print(totalChange)
	return totalChange

#main network training function
def trainNetwork(Max_iters = 1):
	predictionMatrix = []
	for i in range(len(dataset)):
		predictionMatrix[i] = prediction(dataset[i]);
		for predicted in range(len(dataset[i])):
			if predictionMatrix[i][predicted] != dataset[i][predicted]:
				


		print("This is the error: " + str(error(prediction(dataset[i]),dataset[i])))

trainNetwork()
weightChange(0,1,0,1)
