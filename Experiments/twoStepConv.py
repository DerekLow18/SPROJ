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
dataset = np.genfromtxt('./Downsampled Spikes/pop10/01downsample.csv', delimiter = ',')
datasetWeights = np.genfromtxt('./Syn Weights/groundTruth10.csv', delimiter = ',')
'''
old_stdout = sys.stdout

log_file = open("message.log","w")
sys.stdout = log_file
print(dataset.shape)
print(dataset)

sys.stdout = old_stdout

log_file.close()
'''
#dataset = dataset[90:92] #1,2
#dataset = dataset[15:17] #1,1
#dataset = dataset[23:25] #0,0
#dataset = dataset[597:599] #2,2
#dataset = dataset[231:233] #2,0
#dataset = dataset[234:236] #0,1
#dataset = dataset[499:501] #0,2
#dataset = dataset[556:558] #2,1
#dataset = dataset[980:982] #0,3
dataset = dataset[981:983] #3,0



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
sigmoidLR = 0.2

def squaredError(prediction,actual):
	squaredErrorVector = []
	for index in range(len(prediction)):
		squaredErrorVector.append((1/2)*(actual[index] - prediction[index])**2)
	return np.sum(squaredErrorVector)


sigmoidSteepness = np.full(dataset.shape[1],10)
print(sigmoidSteepness)
sigmoidCenter = np.full(dataset.shape[1],0.5)
#formula for the prediction of what the next step will look like.
#Currently, it's at sigmoid function
def activation(activity,index):
	global sigmoidSteepness, sigmoidCenter
	return round(1 / (1 + math.exp(-sigmoidSteepness[index] * (activity-sigmoidCenter[index]))),9)

def pdSquaredError(predicted, actual):
	return round(-(actual - predicted),9)

def pdEuclideanDistance(predicted,actual):
#calculate value for partial deriv of euclidean distance w.r.t. predicted
	return (predicted-actual)/(np.sqrt((predicted-actual)**2))

#partial derivative of the activation function
def pdSigmoid(x,index):
	global sigmoidSteepness,sigmoidCenter
	numerator = sigmoidSteepness[index]*np.exp(-(sigmoidSteepness[index])*(x - sigmoidCenter[index]))
	denominator = (1 + np.exp(-(sigmoidSteepness[index])*(x-sigmoidCenter[index])))**2
	return numerator/denominator
	#return round(x*(1-x),9)

'''
The next to functions are used to update the constants of the sigmoid
'''
#partial derivative of the activation function WRT steepness
def pdSigmoidWRTSteepness(x,index):
	global sigmoidSteepness, sigmoidCenter
	numerator = np.exp(-(sigmoidSteepness[index])*(x-sigmoidCenter[index]))*(sigmoidCenter[index]-x)
	denominator = (1 + np.exp(-(sigmoidSteepness[index])*(x-sigmoidCenter[index])))**2
	return numerator/denominator

#partial derivative of the activation funcion WRT shift
def pdSigmoidWRTShift(x,index):
	global sigmoidSteepness, sigmoidCenter
	numerator = np.exp(-(sigmoidSteepness[index])*(x-sigmoidCenter[index]))*(sigmoidSteepness[index])
	denominator = (1 + np.exp(-(sigmoidSteepness[index])*(x-sigmoidCenter[index])))**2
	return numerator/denominator
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
		adjustedStep[value] = activation(adjustedStep[value],value)
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
def weightChangeOutput(predicted,actual,priorStep,index):
	i = round(pdSquaredError(predicted,actual),9)
	j = round(pdSigmoid(predicted,index),9)
	#print(j)
	#partial derivative of activation function with respect to the activity
	totalChange = round(i*j*priorStep,9)
	#totalChange = predicted - actual
	return totalChange

def sigmoidChangeOutput(predicted,actual,index):
	i = round(pdSquaredError(predicted,actual),9)
	j = round(pdSigmoidWRTShift(predicted,index),9)
	k = round(pdSigmoidWRTSteepness(predicted,index),9)
	return i*j, i*k
#store the errors 
#main network training function
def trainNetworkOneStep(timestep, predictionSet, Max_iters = 1,data = dataset):
	i=0
	global weights, sigmoidSteepness, sigmoidCenter
	while (i <Max_iters):
		predictionMatrix = predictionSet;#store the predictions for the array into a matrix
		'''
		now that we have the predictions, we need to calculate the weight change for each weight in the
		weight matrix. Start with the output layer's weights from the hidden layer
		'''
		#updatedWeights = copy.deepcopy(weights)
		updatedWeights = np.zeros((dataset.shape[1],dataset.shape[1]))
		updatedSteep = np.zeros(sigmoidSteepness.shape[0])
		updatedCenter = np.zeros(sigmoidCenter.shape[0])
		#weightArrayIndex is the index of the source
		for weightArrayIndex in range(len(weights)):
			#weightValue Index is the index of the sink
			for weightValueIndex in range(len(weights[weightArrayIndex])):

				#defining some variables here
				weightValue = weights[weightArrayIndex][weightValueIndex]
				predicted = predictionMatrix[weightValueIndex]
				actual = data[timestep+1][weightValueIndex]
				priorStep = data[timestep][weightArrayIndex]

				'''#4/21/18
				#calculate the sigmoid change
				sigmoidShift, sigmoidSteep = sigmoidChangeOutput(predicted,actual,weightValueIndex)
				updatedSteep[weightValueIndex] = round(sigmoidSteepness[weightValueIndex] - learningRate*sigmoidSteep,9)
				updatedCenter[weightValueIndex] = round(sigmoidCenter[weightValueIndex] - sigmoidLR*sigmoidShift,9)
				'''
				'''
				sigmoidSteepness = updatedSteep
				sigmoidCenter = updatedCenter
				'''
				#calculate weight change for each weight, where first param is outputArray, second is the actual array, and third is the output from the prior step
				weightDelta=weightChangeOutput(predicted,actual,priorStep,weightValueIndex)
				updatedWeights[weightArrayIndex][weightValueIndex] = round(weightValue - learningRate*weightDelta,9)

		#4/21/18
		for weightValueIndex in range(len(sigmoidSteepness)):
			predicted = predictionMatrix[weightValueIndex]
			actual = data[timestep+1][weightValueIndex]

			sigmoidShift, sigmoidSteep = sigmoidChangeOutput(predicted,actual,weightValueIndex)
			updatedSteep[weightValueIndex] = round(sigmoidSteepness[weightValueIndex] - learningRate*sigmoidSteep,9)
			updatedCenter[weightValueIndex] = round(sigmoidCenter[weightValueIndex] + sigmoidLR*sigmoidShift,9)

		i += 1
	weights = updatedWeights
	sigmoidSteepness = updatedSteep
	sigmoidCenter = updatedCenter


def trainNetwork(Max_iters = 100):
	global weights
	priorMSE = 100
	j = 0
	old_stdout = sys.stdout

	log_file = open("message.log","w")
	sys.stdout = log_file
	while (j<Max_iters):
		predictedMatrix = []
		#training step, change the weights
		'''
		old_stdout = sys.stdout

		log_file = open("message.log","w")
		sys.stdout = log_file
		'''
		for i in range(0,len(dataset)-1):
			predictionTimeStep = prediction(dataset[i])
			#predictedMatrix.append(predictionTimeStep.round())
			trainNetworkOneStep(i, predictionTimeStep)
			#print(((dataset[i] - predictionTimeStep) ** 2).mean(axis=None))
			#print((((datasetWeights - weights) ** 2).mean(axis=None)))

		#sys.stdout = old_stdout
		#log_file.close()

		#create a new predicted matrix, from the weights of the previous iteration
		for i in range(len(dataset)-1):
			predictedMatrix.append(prediction(dataset[i]))
		
		#calculate the mean squared error
		mse = ((dataset[:len(dataset)-1] - predictedMatrix) ** 2).mean(axis=None)
		#print("Curr diff: ", abs(priorMSE - mse))
		#print("target diff:", 0.0005*priorMSE)
		#print("Prior MSE: ",priorMSE,"\n")
		#print("MSE: ",mse,"\n")
		print(mse)
		#calculate the mean squared error weights
		#weightMse = ((datasetWeights - weights) ** 2).mean(axis=None)
		#print("weightMSE:",weightMse)
		if mse == 0:
			break
			'''
		if abs(priorMSE - mse) <= 0.0005*(priorMSE):
			break
			'''
			
		#print("After:\n",weights,"\n")

		#check to compare previous error to current error. If close enough, break
		'''
		if j%10 == 0:
			print(weights)
			print(j)
		'''
		priorMSE = mse
		j += 1
	sys.stdout = old_stdout
	log_file.close()
	print(j)
	return predictedMatrix


print("Before: \n",weights,"\n")
x = trainNetwork()
x = np.array(x)

print("After:\n",weights,"\n")
print("final output is ",x)
print(dataset)
print("sigmoid params \n",sigmoidSteepness,"\n",sigmoidCenter)
'''
np.savetxt("./Final Results/pop10/resultingMatrix1.csv",weights,delimiter=",")
np.savetxt("./Final Results/pop10/finalPrediction.csv",x,delimiter=',')
#normalized results"
xmax, xmin = x.max(), x.min()
normX = (x - xmin)/(xmax - xmin)
np.savetxt("./Final Results/pop10/normalizedFinalPrediction.csv",normX,delimiter = ',')

threshX = np.where(normX > 0.8, 1, 0)
np.savetxt("./Final Results/pop10/thresholdedFinalPrediction.csv",threshX,delimiter = ',')
#normalized weights
weightMax, weightMin = weights.max()+abs(0.1*weights.max()), weights.min() - abs(0.001*weights.min())
normWeights = (weights-weightMin)/(weightMax-weightMin)
np.savetxt("./Final Results/pop10/normalizedFinalWeights.csv",normWeights,delimiter=',')
threshIndex=0
while threshIndex <= 1:
	print(threshIndex)
	threshX = np.where(normWeights > threshIndex, 1, 0)
	np.savetxt("./varSig thresholds/%dweightMatrix.csv" % (threshIndex*100),threshX,delimiter = ',')
	threshIndex += 0.01
'''