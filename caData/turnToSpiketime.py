import sys,os
import numpy as np
import fnmatch

#read the csv file. Each column from c onwards is a distinct neuron id.
def convertToBinary(fileName):
	rawSpikeTimes = np.genfromtxt(fileName,delimiter=',')
	times = rawSpikeTimes[1]
	newMatrix = np.zeros((rawSpikeTimes.shape[0],rawSpikeTimes.shape[1]-2))
	print(rawSpikeTimes.shape)
	print(newMatrix.shape)
	for i in range(len(rawSpikeTimes)):
		for j in range(2,len(rawSpikeTimes[i])):
		 if rawSpikeTimes[i][j] != 0:
		 	newMatrix[i][j-2] = 1

	return newMatrix
#first, I need to parse the entire directory and subdirectories
#path = "./_caimg s49 csv - spiking only and good files only/"
path = "./_caimg s46 csv - only good experiments and only spiking/"
for subdir in os.listdir(path):
	for file in os.listdir(path+subdir):
		if fnmatch.fnmatch(file,"*S.txt"):
			fileID = os.path.splitext(file)[0]
			conversion = convertToBinary(path+subdir+"/"+file)
			np.savetxt("./binaryData46/"+fileID+"converted.csv",conversion,delimiter = ',')
'''
conversion = convertToBinary("./_caimg s46 csv - only good experiments and only spiking/140516/140516S.txt")
np.savetxt("./binaryData/ex.csv",conversion, delimiter = ',')
'''