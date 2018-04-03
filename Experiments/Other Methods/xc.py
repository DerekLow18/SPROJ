from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cch,covariance, corrcoef
from elephant.spike_train_generation import homogeneous_poisson_process
from quantities import Hz, s, ms
import matplotlib.pyplot as plt
import numpy as np
import neo

'''
Cross-Correlation of spike trains to form a correlation matrix
'''
def spikeTimeToArrays(dataset):
	#convert a spiketime matrix to an array of neurons, and spiketimes within those arrays
	dataset = np.genfromtxt(dataset,delimiter = ',')
	idMatrix = [[] for i in range(len(dataset))]#np.zeros((np.shape(dataset)[0],1))
	for i in range(len(dataset)):
		for j in range(len(dataset[i])):
			if dataset[i][j] == 1:
				print(i,j)
				idMatrix[i].append(j)
	print(idMatrix)
	return idMatrix

dataset = spikeTimeToArrays('../Spike Results/1idTimes.csv')
print(dataset)
neoDataset = []
[neoDataset.append(neo.SpikeTrain(i,units = 'ms',t_start = 0,t_stop = 10000.0)) for i in dataset]
print(neoDataset)

elephantDataset = []
[elephantDataset.append(BinnedSpikeTrain(j,binsize = 10*ms)) for j in neoDataset]
#print(elephantDataset)
'''
st1 = homogeneous_poisson_process(
        rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
st2 = homogeneous_poisson_process(
        rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
print(st1,st2)
#st1 = BinnedSpikeTrain(st1, num_bins = 10, binsize = 1*ms, t_start = 0*s)
#st2 = BinnedSpikeTrain(st2, num_bins = 10, binsize = 1*ms, t_start = 0*s)
print(BinnedSpikeTrain([st1,st2],binsize = 5*ms))
'''
cov_matrix = corrcoef(elephantDataset)
print(cov_matrix)
#plt.hist(cch(st1,st2)[1])
#plt.show()