import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
#import scipy.spatial
import copy
#from sympy import symbols, diff

# convert an array of values into a dataset matrix
'''
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
#now, each array in dataset is representative of a single timestep,
#where each value is whether or not the neuron is spiking at that particular time

#bin the data set so that every set of 10 steps is reshaped to be one step,
#and multiple spikes during that time only counted as 1
dataset = downsample(dataset,10)

print(dataset.shape)
print(dataset)
'''
#making sure it works on a smaller example
dataset =[[0.05,0.10],[0.01,0.99]]
hidden_layer_weights = [[0.15,0.25],[0.20,0.30]]
output_layer_weights = [[0.40,0.50],[0.45,0.55]]
hidden_layer_bias = 0.35
output_layer_bias = 0.6

#intilizalize the weight array
weights = np.random.rand(10,10)
learningRate = 0.5

#error calculation between the predicted step and the actual step, euclidean distance
def error(prediction, actual):
	return scipy.spatial.distance.euclidean(prediction, actual)

def squaredError(prediction,actual):
	squaredErrorVector = []
	for index in range(len(prediction)):
		squaredErrorVector.append((1/2)*(actual[i] - prediction[i])**2)
	return squaredErrorVector

#formula for the prediction of what the next step will look like.
#Currently, it's at sigmoind function
def activation(activity):
	return round(1 / (1 + math.exp(-activity)),9)

def pdSquaredError(predicted, actual):
	return round(-(actual - predicted),9)

def pdEuclideanDistance(predicted,actual):
#calculate value for partial deriv of euclidean distance w.r.t. predicted
	return (predicted-actual)/(np.sqrt((predicted-actual)**2))

#partial derivative of the activation function
def pdSigmoid(x):
	return round(x*(1-x),9)

#using this for the purposes of testing a standard prediction formula
def prediction(timeStep):
	#first calculate the hidden layer values, called prediction array
	predictionArray = [0,0]
	predictionActivity = [0,0]
	for hiddenIndex in range(len(predictionArray)):
		predicted = 0
		for inputIndex in range(len(dataset[0])):
			predicted += dataset[0][inputIndex]*hidden_layer_weights[inputIndex][hiddenIndex]
		predicted = predicted + hidden_layer_bias
		predictionActivity[hiddenIndex] = round(predicted,9)
		predictionArray[hiddenIndex] = round(activation(predicted),9)
	'''
	now we calculate the values for the output array
	'''
	outputArray = [0,0]
	outputActivity = [0,0]
	for outputIndex in range(len(outputArray)):
		predicted = 0
		for hiddenIndex in range(len(predictionArray)):
			predicted += predictionArray[hiddenIndex]*output_layer_weights[hiddenIndex][outputIndex]
		predicted = predicted + output_layer_bias
		outputActivity[outputIndex] = round(predicted,9)
		outputArray[outputIndex] = round(activation(predicted),9)

	#print(predictionArray)
	#print(outputArray)
	return [predictionArray, outputArray, predictionActivity, outputActivity]

#takes a timeStep and attempts to predict the next time step
'''
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

'''
change the weight between one source neuron and the target neuron

'''
def weightChangeOutput(predicted,actual,priorStep):
	'''x, y= symbols('x y', real=True)
	f = ((x-y)**2)**(1/2)
	deriv = diff(f, x)
	print(deriv)'''
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
'''
calculate the weight change for one weight in the hidden layer

Prior step is the input to the hidden layer
'''
def weightChangeHidden(predictedArray,actualArray,priorStep,weightIndex,hiddenOutput):
	totalError = 0
	#here, predictedArray is the array from the layer forward to the current hidden layer
	#actual array is the corresponding output array
	#print("THe predicted array is ",predictedArray)
	#print("the actual array is ", actualArray)
	for outputIndex in range(len(predictedArray)):
		i = round(pdSquaredError(predictedArray[outputIndex],actualArray[outputIndex]),9)
		j = round(pdSigmoid(predictedArray[outputIndex]),9)
		w = output_layer_weights[weightIndex][outputIndex]
		#print(i,j,w)
		totalError = totalError + round(i*j*w,9)
	hO = round(pdSigmoid(hiddenOutput),9)
	#print(totalError,hO,priorStep)
	totalChange = round(totalError*hO*priorStep,9)
	return totalChange

#main network training function
def trainNetwork(Max_iters = 10000):
	#predictionMatrix = []
	#Iterates through all values in the data set
	#for i in range(len(dataset)):
		#predict the value for the next step and store it
	i=0
	while (i <Max_iters):
		predictionMatrix = prediction(dataset[0]);#store the predictions for the array into a matrix
		'''
			for predicted in range(len(dataset[i])):
							#check to see if prediction and actual are different
				if predictionMatrix[i][predicted] != dataset[i][predicted]:
					return
					'''
		'''
		now that we have the predictions, we need to calculate the weight change for each weight in the
		weight matrix. Start with the output layer's weights from the hidden layer
		'''
		global output_layer_weights,hidden_layer_weights
		#print("Before: \n", output_layer_weights,"\n",hidden_layer_weights,"\n")
		updatedWeights = copy.deepcopy(output_layer_weights)
		for weightArrayIndex in range(len(output_layer_weights)):
			for weightValueIndex in range(len(output_layer_weights[weightArrayIndex])):
				#print("Calculating for ", weightArrayIndex,weightValueIndex)
				weightValue = output_layer_weights[weightArrayIndex][weightValueIndex]
				#calculate weight change for each weight, where first param is outputArray, second is the actual array, and third is the output from the hidden
				weightDelta=weightChangeOutput(predictionMatrix[1][weightValueIndex],dataset[1][weightValueIndex],predictionMatrix[0][weightArrayIndex])
				#print("degree of change is", weightDelta)
				updatedWeights[weightArrayIndex][weightValueIndex] = weightValue - learningRate*weightDelta
		#print(updatedWeights)
		'''
		now we do a similar thing for the input to hidden layer weights.
		'''
		updatedIToHWeights = copy.deepcopy(output_layer_weights)
		for weightArrayIndex in range(len(hidden_layer_weights)):
			for weightValueIndex in range(len(hidden_layer_weights[weightArrayIndex])):
				weightValue = hidden_layer_weights[weightArrayIndex][weightValueIndex]
				weightDelta=weightChangeHidden(predictionMatrix[1],dataset[1],dataset[0][weightArrayIndex],weightArrayIndex,predictionMatrix[0][weightValueIndex])
				updatedIToHWeights[weightArrayIndex][weightValueIndex] = weightValue - learningRate * weightDelta
		#print(updatedIToHWeights)
		output_layer_weights = updatedWeights
		hidden_layer_weights = updatedIToHWeights

		i += 1
	print("After: \n",output_layer_weights,"\n",hidden_layer_weights,"\n")
	print(predictionMatrix[1])
		#print("This is the error: " + str(error(prediction(dataset[i]),dataset[i])))
'''predicted = prediction(dataset[0])
actualArray = dataset[1]
priorStep = dataset[0][0]
hiddenOutput = predicted[0]
print(weightChangeHidden(predicted[1],actualArray,priorStep,0,hiddenOutput[0]))
'''
trainNetwork()
