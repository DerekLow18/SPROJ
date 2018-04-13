import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
#import scipy.spatial
import copy
import sys
np.set_printoptions(threshold=np.nan)
'''
This version of the neural network changes the weights by predicting timestep t+1
based on time t, then taking the dataset, producing a prediction model using the final weights,
and comparing the prediction times to the actual times.

Then, we view the resulting weight matrix as, in some way, informative of the ground truth
population matrix of the dataset.

#now, each array in dataset is representative of a single timestep,
#where each value is whether or not the neuron is spiking at that particular time

#bin the data set so that every set of 10 steps is reshaped to be one step,
#and multiple spikes during that time only counted as 1
dataset = downsample(dataset,10)
'''
#import the dataset
dataset = np.genfromtxt('./Downsampled Spikes/01downsample.csv', delimiter = ',')

'''
old_stdout = sys.stdout

log_file = open("message.log","w")
sys.stdout = log_file
print(dataset.shape)
print(dataset)

sys.stdout = old_stdout

log_file.close()
'''
#dataset = dataset[9:11]
#print(dataset)


#calculate spike rate matrix
spikeRate = np.zeros((dataset.shape[1],1))
for i in range(len(dataset.transpose())):
	for j in dataset.transpose()[i]:
		if j == 1:
			spikeRate[i] = spikeRate[i] + 1
	spikeRate[i] = spikeRate[i]/len(dataset)
print(spikeRate)

#intilizalize the weight array
weights = np.random.rand(dataset.shape[1],dataset.shape[1])
#weights = np.zeros((dataset.shape[1],dataset.shape[1]))
learningRate = 0.5

#error calculation between the predicted step and the actual step, euclidean distance
def error(prediction, actual):
	return scipy.spatial.distance.euclidean(prediction, actual)

def squaredError(prediction,actual):
	squaredErrorVector = []
	for index in range(len(prediction)):
		squaredErrorVector.append((1/2)*(actual[index] - prediction[index])**2)
	return np.sum(squaredErrorVector)


sigmoidSteepness = 10
sigmoidCenter = 0.5
#formula for the prediction of what the next step will look like.
#Currently, it's at sigmoid function
def activation(activity):
	return round(1 / (1 + math.exp(-sigmoidSteepness * (activity-sigmoidCenter))),9)

def pdSquaredError(predicted, actual):
	return round(-(actual - predicted),9)

def pdEuclideanDistance(predicted,actual):
#calculate value for partial deriv of euclidean distance w.r.t. predicted
	return (predicted-actual)/(np.sqrt((predicted-actual)**2))

#partial derivative of the activation function
def pdSigmoid(x):
	
	global sigmoidSteepness,sigmoidCenter
	numerator = sigmoidSteepness*np.exp(-(sigmoidSteepness)*(x - sigmoidCenter))
	denominator = (1 + np.exp(-(sigmoidSteepness)*(x-sigmoidCenter)))**2
	return numerator/denominator
	
	#return round(x*(1-x),9)

#turns out we don't want this, because we want to keep probability of spiking
#separate for each neuron
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#takes a timeStep to predict the next time step

def prediction(timeStep):
	global weights, activation
	#matrix multiply the weight matrix with the spiking matrix
	adjustedStep = np.matmul(timeStep, weights)
	#go through all values of the adjusted step matrix,
	#and multiply them by the activation function
	for value in range(len(adjustedStep)):
		adjustedStep[value] = activation(adjustedStep[value])
		#multiplication with spike rate
		#adjustedStep[value] = adjustedStep[value]*spikeRate[value]
	#return the resulting and final adjusted step
	#print(adjustedStep)
	return adjustedStep
	
def twoTimeInputPrediction(timeStep,timeStep1):
	#take the time step, and the next time step, to calculate the third time step
	return (prediction(timeStep) + prediction(timeStep1))/2

'''
change the weight between one source neuron and the target neuron

'''
def weightChangeOutput(predicted,actual,priorStep):
	i = round(pdSquaredError(predicted,actual),9)
	j = round(pdSigmoid(predicted),9)
	#partial derivative of activation function with respect to the activity
	totalChange = round(i*j*priorStep,9)
	#totalChange = predicted - actual
	return totalChange

#main network training function
def trainNetworkOneStep(timestep, predictionSet, Max_iters = 1,data = dataset):
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
				weightDelta=weightChangeOutput(predicted,actual,priorStep)

				updatedWeights[weightArrayIndex][weightValueIndex] = round(weightValue - learningRate*weightDelta,9)

		i += 1
	weights = updatedWeights

def trainNetwork(Max_iters = 10):
	global weights
	priorMSE = 100
	j = 0
	while (j<Max_iters):
		predictedMatrix = []
		#training step, change the weights
		for i in range(0,len(dataset)-1):
			predictionTimeStep = prediction(dataset[i])
			#predictedMatrix.append(predictionTimeStep.round())
			trainNetworkOneStep(i, predictionTimeStep)
		
		#create a new predicted matrix, from the weights of the previous iteration
		for i in range(len(dataset)-1):
			predictedMatrix.append(prediction(dataset[i]))
		
		#calculate the mean squared error
		mse = ((dataset[:len(dataset)-1] - predictedMatrix) ** 2).mean(axis=None)
		print("Curr diff: ", abs(priorMSE - mse))
		print("target diff:", 0.0005*priorMSE)
		print("Prior MSE: ",priorMSE,"\n")
		print("MSE: ",mse,"\n")
		
		if mse == 0:
			break
			'''
		if abs(priorMSE - mse) <= 0.0005*(priorMSE):
			break
			'''
			
		#print("After:\n",weights,"\n")

		#check to compare previous error to current error. If close enough, break
		if j%10 == 0:
			print(weights)
			print(j)
		priorMSE = mse
		j += 1

	return predictedMatrix


print("Before: \n",weights,"\n")
x = trainNetwork()
x = np.array(x)
print("After:\n",weights,"\n")
print("final output is ",x)
print(dataset)
np.savetxt("resultingMatrix1.csv",weights,delimiter=",")
np.savetxt("finalPrediction.csv",x,delimiter=',')
#normalized results"
xmax, xmin = x.max(), x.min()
normX = (x - xmin)/(xmax - xmin)
np.savetxt("normalizedFinalPrediction.csv",normX,delimiter = ',')

threshX = np.where(normX > 0.5, 1, 0)

np.savetxt("thresholdedFinalPrediction.csv",threshX,delimiter = ',')

