import numpy as np
import sys

#calculate spike rate matrix
def spikeRate(spikeTime):
	spikeTime = np.genfromtxt(spikeTime, delimiter = ',')
	spikeRate = np.zeros((spikeTime.shape[1],1))
	for i in range(len(spikeTime.transpose())):
		for j in spikeTime.transpose()[i]:
			if j == 1:
				spikeRate[i] = spikeRate[i] + 1
		spikeRate[i] = spikeRate[i]/len(spikeTime)
	return spikeRate

if __name__=='__main__':
	if len(sys.argv) < 2:
		raise Exception("Please enter a csv to calculate spike rate of.")
		exit()
	spikeTime = sys.argv[1]
	print(spikeRate(spikeTime))
