import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
#import scipy.spatial
import copy
import sys
np.set_printoptions(threshold=np.nan)

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
dataset = np.transpose(dataset)
print(dataset.shape)
#now, each array in dataset is representative of a single timestep,
#where each value is whether or not the neuron is spiking at that particular time

#bin the data set so that every set of 10 steps is reshaped to be one step,
#and multiple spikes during that time only counted as 1
dataset = downsample(dataset,10)

old_stdout = sys.stdout

log_file = open("message.log","w")
sys.stdout = log_file
print(dataset.shape)
print(dataset)

sys.stdout = old_stdout

log_file.close()

#intilizalize the weight array
weights = np.random.rand(dataset.shape[1],dataset.shape[1])
learningRate = 0.5

#error calculation between the predicted step and the actual step, euclidean distance
def error(prediction, actual):
	return scipy.spatial.distance.euclidean(prediction, actual)

def squaredError(prediction,actual):
	squaredErrorVector = []
	for index in range(len(prediction)):
		squaredErrorVector.append((1/2)*(actual[index] - prediction[index])**2)
	return np.sum(squaredErrorVector)

#formula for the prediction of what the next step will look like.
#Currently, it's at sigmoid function
def activation(activity):
	'''
	if activity > 1:
		return 1
	else:
		return round(activity)
		'''
	return round(1 / (1 + math.exp(-5 * (activity-0.5))),9)
	#return round(1 / (1 + math.exp(-activity)),9)

def pdSquaredError(predicted, actual):
	return round(-(actual - predicted),9)

def pdEuclideanDistance(predicted,actual):
#calculate value for partial deriv of euclidean distance w.r.t. predicted
	return (predicted-actual)/(np.sqrt((predicted-actual)**2))

#partial derivative of the activation function
def pdSigmoid(x):
	return round(x*(1-x),9)

#takes a timeStep and attempts to predict the next time step

def prediction(timeStep):
	global weights, activation
	#matrix multiply the weight matrix with the spiking matrix
	adjustedStep = np.matmul(timeStep, weights)
	#go through all values of the adjusted step matrix,
		#and multiply them by the activation function
	for value in range(len(adjustedStep)):
		adjustedStep[value] = activation(adjustedStep[value])
		#print(adjustedStep)
	#return the resulting and final adjusted step
	return adjustedStep
	

'''
change the weight between one source neuron and the target neuron

'''
def weightChangeOutput(predicted,actual,priorStep):
	#print("priorStep is",priorStep)
	i = round(pdSquaredError(predicted,actual),9)
	#print("the pd squared error is ", i)
	#partial derivative of euclidean distance with respect to the prediction
	#print("the activity is", predicted)
	j = round(pdSigmoid(predicted),9)
	#print("the pd sigmoid is",j)
		#partial derivative of activation function with respect to the activity
	totalChange = round(i*j*priorStep,9)
	return totalChange

#main network training function
def trainNetworkOneStep(timestep, predictionSet, Max_iters = 1,data = dataset):
	#predictionMatrix = []
	#Iterates through all values in the data set
	#for i in range(len(dataset)):
		#predict the value for the next step and store it
	i=0
	global weights
	while (i <Max_iters):
		predictionMatrix = predictionSet;#store the predictions for the array into a matrix
		'''
		now that we have the predictions, we need to calculate the weight change for each weight in the
		weight matrix. Start with the output layer's weights from the hidden layer
		'''
		#updatedWeights = copy.deepcopy(weights)
		updatedWeights = np.zeros((dataset.shape[1],dataset.shape[1]))
		for weightArrayIndex in range(len(weights)):
			for weightValueIndex in range(len(weights[weightArrayIndex])):

				#defining some variables here
				weightValue = weights[weightArrayIndex][weightValueIndex]
				predicted = predictionMatrix[weightValueIndex]
				actual = data[timestep+1][weightValueIndex]
				priorStep = data[timestep][weightArrayIndex]

				#calculate weight change for each weight, where first param is outputArray, second is the actual array, and third is the output from the prior step
				weightDelta=weightChangeOutput(predictionMatrix[weightValueIndex],data[timestep+1][weightValueIndex],data[timestep][weightArrayIndex])

				updatedWeights[weightArrayIndex][weightValueIndex] = round(weightValue - learningRate*weightDelta,9)

		#print(updatedWeights)
		#print(squaredError(predictionMatrix[1],dataset[1]))

		i += 1
	weights = updatedWeights

def trainNetwork(Max_iters = 1):
	global weights
	priorMSE = 100
	predictedMatrix = []
	j = 0
	while (j<Max_iters):
		for i in range(len(dataset)-1):
			predictionTimeStep = prediction(dataset[i])
			predictedMatrix.append(predictionTimeStep)

			mse = ((dataset[i] - predictionTimeStep) ** 2).mean(axis=None)
			while priorMSE - mse > 0.05*priorMSE:
				if round(priorMSE,9) == round(mse,9):
					break
				trainNetworkOneStep(i, predictionTimeStep)
				priorMSE = mse
				print("iteration",i)
				#print(weights)
				print(mse)
			
			print(dataset[i])
			print(predictionTimeStep)

			mse = ((dataset[i] - predictionTimeStep) ** 2).mean(axis=None)
			print("After:\n",weights,"\n")
			print("MSE: ",mse,"\n")
			priorMSE = mse
		i += 1


print("Before: \n",weights,"\n")
trainNetwork()
print("After:\n",weights,"\n")
np.savetxt("resultingMatrix1.csv",weights,delimiter=",")