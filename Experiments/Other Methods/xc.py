from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cch,covariance, corrcoef, cross_correlation_histogram
from elephant.spike_train_generation import homogeneous_poisson_process
from quantities import Hz, s, ms
import matplotlib.pyplot as plt
import numpy as np
import neo
import sys, os, fnmatch, re

'''
Cross-Correlation of spike trains to form a correlation matrix
'''
def spikeTimeToArrays(dataset):
	#convert a spiketime matrix to an array of neurons, and spiketimes within those arrays
	dataset = np.genfromtxt(dataset,delimiter = ',')
	dataset = dataset.transpose()
	idMatrix = [[] for i in range(len(dataset))]#np.zeros((np.shape(dataset)[0],1))
	for i in range(len(dataset)):
		for j in range(len(dataset[i])):
			if dataset[i][j] == 1:
				idMatrix[i].append(j)
	return idMatrix


dataset = spikeTimeToArrays('../Downsampled Spikes/01downsample.csv')
#dataset = dataset.transpose()
#print(dataset)
dataSpikeTimes = np.genfromtxt('../Downsampled Spikes/01downsample.csv', delimiter = ',')
dataSpikeTimes = dataSpikeTimes.transpose()
#print(dataset)
neoDataset = []
[neoDataset.append(neo.SpikeTrain(i,units = 'ms',t_start = 0,t_stop = 1000.0)) for i in dataset]
print(neoDataset[0])
print(neoDataset[1])
'''
elephantDataset = []
[elephantDataset.append(BinnedSpikeTrain(j,binsize = 10*ms)) for j in neoDataset]
x = BinnedSpikeTrain(neoDataset,binsize=10*ms)

cov_matrix = corrcoef(x,binary=True)

'''
#pyplot xcorr returns an array of the timelag used, and a corrsponding array
#of the calculated correlation coefficient between the two spike trains,
#according to that timelag
#print("pyplot Correlate:\n",plt.xcorr(dataSpikeTimes[0],dataSpikeTimes[1]))

'''
Loop through every combination, and generate the cross_correlograms. Save the highest value
bins to identify the strongest connections in the network.

We want to consider x amount of the highest to be the connections

Keep a dictionary with {i:j,n}, where the connection of i to j has n highest values in a bin
'''
#first, we calculate cch for neurons i,j and return the maximum 

def calcCCH(i,j,dataset):

	#use elephant to recreate cross_correlation_histograms
	cch = cross_correlation_histogram((BinnedSpikeTrain(neoDataset[i],binsize=1*ms)),
		(BinnedSpikeTrain(neoDataset[j],binsize=1*ms)),window=[-10,10],border_correction = True,
		binary = False, kernel = None)
	cchArray1 = np.array(cch[0][:,0].magnitude)
	return cchArray1.max()

def generateCorrelationMatrix(neoDataset,connectionDict):
	connectionDictionary = connectionDict
	for i in range(len(neoDataset)):
		for j in range(len(neoDataset)):
			if i != j:
				connectionDictionary[i][j] = connectionDictionary[i][j] + calcCCH(i,j,neoDataset)
	return connectionDictionary

#This code will produce a single cross correlogram

cch = cross_correlation_histogram((BinnedSpikeTrain(neoDataset[0],binsize=1*ms)),
	(BinnedSpikeTrain(neoDataset[0],binsize=1*ms)),window=[-50,50],border_correction = True,
	binary = False, kernel = None)
cchArray =  cch[0][:,0].magnitude
cchArrayTime = cch[0].times.magnitude
cchArrayNP = np.array(cchArray)
print("argmax is:",cchArrayNP.max())


#calculate the cross-correlograms of the entire dataset,
#produce the corresponding correlation matrix
'''
print(calcCCH(8,9,neoDataset))
correlationArray = generateCorrelationMatrix(neoDataset)
print(correlationArray)
'''

#for i in cchArray:
	#print(i)
#print(cchArray)
'''
print(len(cchArray))
plt.bar(cchArrayTime,cchArray,cch[0].sampling_period.magnitude)
plt.show()
'''

'''
Can we reconstruct the network purely based on xc?
Do this cross-correlation procedure, and add for all samples in the dataset

Average the cross-correlation values by the number of datasets observed
'''

if __name__ == "__main__":

	#initalize the correlation matrix
	data = np.genfromtxt('../Downsampled Spikes/01downsample.csv', delimiter = ',')
	correlationMatrix = np.zeros((len(data.transpose()),len(data.transpose())))
	numFiles = 0
	for file in os.listdir("../Downsampled Spikes/"):
		if fnmatch.fnmatch(file,'*downsample.csv'):
			numFiles += 1
			dataset = spikeTimeToArrays("../Downsampled Spikes/"+file)
			neoDataset = []
			[neoDataset.append(neo.SpikeTrain(i,units = 'ms',t_start = 0,t_stop = 1000.0)) for i in dataset]
			print("calculating for:",file)
			correlationMatrix = generateCorrelationMatrix(neoDataset,correlationMatrix)
			print(correlationMatrix)

	#divide the values by the number of datasets observed
	avgMatrix = correlationMatrix/numFiles

	#find the minimum and maximum values, for normalization
	minvalue = np.min(avgMatrix[np.nonzero(avgMatrix)])
	maxvalue = avgMatrix.max()
	#normalize the matrix via non-zero min and max
	normX = (avgMatrix - minvalue)/(maxvalue - minvalue)
	#replace the diagonals with 0s again
	np.fill_diagonal(normX,0)
	#threshold to observe
	threshIndex = 0
	#change the threshold for the purpose of ROC
	while threshIndex <= 1:
		print(threshIndex)
		threshX = np.where(normX > threshIndex, 1, 0)
		np.savetxt("./xcThresholds/%dxcMatrix.csv" % (threshIndex*100),threshX,delimiter = ',')
		threshIndex += 0.01
	print(minvalue)
	print(avgMatrix)
	print(normX)
	print(threshX)
