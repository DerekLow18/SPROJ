import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
#import scipy.spatial
import copy
import sys,os,ntpath
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
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

if len(sys.argv) < 2:
	raise Exception("Please enter a spike time csv")
	exit()

dataset = np.genfromtxt(str(sys.argv[1]), delimiter = ',')
fileID = os.path.splitext(path_leaf(str(sys.argv[1])))[0]
'''
old_stdout = sys.stdout

log_file = open("message.log","w")
sys.stdout = log_file
print(dataset.shape)
print(dataset)

sys.stdout = old_stdout

log_file.close()
'''
#dataset = dataset[15:17]
#dataset = dataset[90:92] #1,2
#dataset = dataset[15:17] #1,1
#dataset = dataset[23:25] #0,0
#dataset = dataset[597:599] #2,2
#dataset = dataset[231:233] #2,0
#dataset = dataset[234:236] #0,1
#dataset = dataset[499:501] #0,2
#dataset = dataset[556:558] #2,1
#dataset = dataset[980:982] #0,3
#dataset = dataset[981:983] #3,0
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
learningRate = 0.2

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
def activation(activity):
	global sigmoidSteepness, sigmoidCenter
	print("shapes are:",sigmoidSteepness.shape,activity.shape)
	return (1/(1 + np.exp(np.multiply(-sigmoidSteepness,(activity-sigmoidCenter)))))
	#return round(1 / (1 + math.exp(-sigmoidSteepness * (activity-sigmoidCenter))),9)

def pdSquaredError(predicted, actual):
	return -(actual - predicted)

def pdEuclideanDistance(predicted,actual):
#calculate value for partial deriv of euclidean distance w.r.t. predicted
	return (predicted-actual)/(np.sqrt((predicted-actual)**2))

#partial derivative of the activation function
def pdSigmoid(x,index):
	
	global sigmoidSteepness,sigmoidCenter
	numerator = sigmoidSteepness[0]*np.exp(-(sigmoidSteepness[0])*(x - sigmoidCenter[0]))
	denominator = (1 + np.exp(-(sigmoidSteepness[0])*(x-sigmoidCenter[0])))**2
	return numerator/denominator
	
	#return round(x*(1-x),9)
#takes a timeStep to predict the next time step

def prediction(timeStep):
	global weights, activation
	#matrix multiply the weight matrix with the spiking matrix
	adjustedStep = np.matmul(timeStep, weights)
	#go through all values of the adjusted step matrix,
	#and multiply them by the activation function
	adjustedStep = activation(adjustedStep)
	#return the resulting and final adjusted step
	#print(adjustedStep)
	return adjustedStep
'''
change the weight between one source neuron and the target neuron

'''
def weightChangeOutput(predicted,actual,priorStep,index):
	i = pdSquaredError(predicted,actual)
	j = pdSigmoid(predicted,index)
	#partial derivative of activation function with respect to the activity
	totalChange = i*j*priorStep
	#totalChange = predicted - actual
	return totalChange

#main network training function
def trainNetworkOneStep(timestep, predictionSet, Max_iters = 1,data = dataset):
	i=0
	global weights
	while (i <Max_iters):
		predictionMatrix = predictionSet;#sstore the predictions for the array into a matrix
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
				weightDelta=weightChangeOutput(predicted,actual,priorStep,weightValueIndex)
				#print(weightValue,weightDelta)
				updatedWeights[weightArrayIndex][weightValueIndex] = weightValue - learningRate*weightDelta

		i += 1
	weights = updatedWeights

def trainNetwork(Max_iters = 1):
	global weights
	priorMSE = 100
	j = 0
	'''
	old_stdout = sys.stdout

	log_file = open("message.log","w")
	sys.stdout = log_file
	'''
	while (j<Max_iters):
		predictedMatrix = []
		#training step, change the weights
		for i in range(0,len(dataset)-1):
			predictionTimeStep = prediction(dataset[i])
			#predictedMatrix.append(predictionTimeStep.round())
			trainNetworkOneStep(i, predictionTimeStep)
			print("trained timestep:",i)
		
		#create a new predicted matrix, from the weights of the previous iteration
		for i in range(len(dataset)-1):
			predictedMatrix.append(prediction(dataset[i]))
		
		#calculate the mean squared error
		mse = ((1/2)*(dataset[:len(dataset)-1] - predictedMatrix) ** 2).mean(axis=None)
		#print(mse)
		'''
		print("Curr diff: ", abs(priorMSE - mse))
		print("target diff:", 0.0005*priorMSE)
		print("Prior MSE: ",priorMSE,"\n")
		print("MSE: ",mse,"\n")
		'''
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
	'''
	sys.stdout = old_stdout
	log_file.close()
	'''
	return predictedMatrix

print("Before: \n",weights,"\n")
x = trainNetwork()
x = np.array(x)
print("After:\n",weights,"\n")
print("final output is ",x)
print(dataset)
print("sigmoid params \n",sigmoidSteepness,"\n",sigmoidCenter)
resultsPath = "./Final Results/generalizedCOE/pop50sf/"+fileID+"/"
if not os.path.exists(resultsPath):
	os.mkdir(resultsPath)
np.savetxt(resultsPath+fileID+"resultingMatrix1.csv",weights,delimiter=",")
np.savetxt(resultsPath+fileID+"finalPrediction.csv",x,delimiter=',')
#normalized results"
xmax, xmin = x.max(), x.min()
normX = (x - xmin)/(xmax - xmin)
np.savetxt(resultsPath+fileID+"normalizedFinalPrediction.csv",normX,delimiter = ',')

threshX = np.where(normX > 0.8, 1, 0)
np.savetxt(resultsPath+fileID+"thresholdedFinalPrediction.csv",threshX,delimiter = ',')
#normalized weights
weightMax, weightMin = weights.max()+abs(0.1*weights.max()), weights.min() - abs(0.001*weights.min())
normWeights = (weights-weightMin)/(weightMax-weightMin)
np.savetxt(resultsPath+fileID+"normalizedFinalWeights.csv",normWeights,delimiter=',')
threshIndex=0
while threshIndex <= 1:
	print(threshIndex)
	threshX = np.where(normWeights > threshIndex, 1, 0)
	path = "./generalizedCOE thresholds/pop50sf/"+fileID+"/"
	if not os.path.exists(path):
		os.mkdir(path)
	np.savetxt("./generalizedCOE thresholds/pop50sf/"+fileID+"/%dweightMatrix.csv" % (threshIndex*100),threshX,delimiter = ',')
	threshIndex += 0.01
