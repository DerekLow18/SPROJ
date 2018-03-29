import numpy as np
import matplotlib.pyplot as plot
import sys

def spikeTimeToArrays(dataset):
	#convert a spiketime matrix to an array of neurons, and spiketimes within those arrays
	dataset = np.genfromtxt(dataset,delimiter = ',')
	idMatrix = [[] for i in range(len(dataset))]#np.zeros((np.shape(dataset)[0],1))
	print(idMatrix)
	print(np.shape(idMatrix))
	for i in range(len(dataset)):
		for j in range(len(dataset[i])):
			if dataset[i][j] == 1:
				print(i,j)
				idMatrix[i].append(j)
				print(idMatrix)
	print(idMatrix)
	return idMatrix

def createPlot(dataset):

	#dataset = np.swapaxes(dataset,0,1)
	plot.eventplot(dataset,linestyles = 'solid')
	#print(dataset)
	plot.title("Spike raster plot")
	plot.xlabel('Time')
	plot.ylabel('Neuron')
	plot.show()



def main():
	if len(sys.argv) < 2:
		print("Please enter a csv file that you want to rasterize")
		exit()
	dataset = str(sys.argv[1])
	transDataSet = spikeTimeToArrays(dataset)
	createPlot(transDataSet)

main()