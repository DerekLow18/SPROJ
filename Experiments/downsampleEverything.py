import numpy as np
import os
import fnmatch
import re

#downsampling method to reduce the timestep amount
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

#downsample all files in the Spike Results Directory
if __name__ == "__main__":

	for file in os.listdir('./Spike Results/'):
		if fnmatch.fnmatch(file,'*idTimes.csv'):
			fileID = re.findall("(\d+)idTimes.csv",file)
			
			dataset = np.genfromtxt('./Spike Results/'+file,delimiter = ',')
			dataset = np.transpose(dataset)
			newData = downsample(dataset,10)

			np.savetxt("./Downsampled Spikes/"+str(fileID[0]) + "downsample.csv",newData,delimiter = ',')

