import numpy as np
import sys, os, fnmatch

#calculate spike rate matrix
def spikeRate(spikeTime):
	spikeTime = np.genfromtxt(spikeTime, delimiter = ',')
	spikeRate = np.zeros((spikeTime.shape[1],1))
	for i in range(len(spikeTime.transpose())):
		for j in spikeTime.transpose()[i]:
			if j == 0:
				spikeRate[i] = spikeRate[i] + 1
		spikeRate[i] = spikeRate[i]/len(spikeTime)
	return spikeRate
'''
if __name__=='__main__':
	if len(sys.argv) < 2:
		raise Exception("Please enter a csv to calculate spike rate of.")
		exit()
	spikeTime = sys.argv[1]
	print(spikeRate(spikeTime))
'''
spikeRates = []
if __name__=='__main__':
	if len(sys.argv) <2:
		raise Exception("please provide a directory of spike time matrices")
		exit()
	path = str(sys.argv[1])
	for subdir in os.listdir(path):
		for file in os.listdir(path+subdir):
			if fnmatch.fnmatch(file,"*downsamplethresholdedFinalPrediction.csv"):
				print(path+subdir+"/"+file)
				spikeRates.append(spikeRate(str(path+subdir+"/"+file)))

np.savetxt("downsampleResultSpikeRates.csv",np.array(spikeRates),delimiter = ',')

