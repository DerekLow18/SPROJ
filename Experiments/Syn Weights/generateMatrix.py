import numpy
import sys

def createRandomNetwork(file_name, popSize):
	#generates an adj matrix based on parameters of network, and depending on formula
	#such as small world, clustering, criticality, etc. WIP
	adjMatrix = numpy.zeros((popSize, popSize))
	#ensure that there are no self-connections at each neuron
	for ith_neuron_index in range(len(adjMatrix)):
		adjMatrix[ith_neuron_index][ith_neuron_index] = 0

	#fill out the adj matrix for all other neurons
	for ith_neuron_index in range(len(adjMatrix)):
		for jth_neuron_index in range(len(adjMatrix)):
			if ith_neuron_index != jth_neuron_index:
				adjMatrix[ith_neuron_index][jth_neuron_index]= numpy.random.randint(0,high=2)
	print adjMatrix
	numpy.savetxt("./"+file_name, adjMatrix, delimiter=",")
	return

if __name__=="__main__":
	if len(sys.argv) < 2:
		raise Exception("Not enough arguments! Please enter the file name and population size!")
		exit()
	createRandomNetwork(sys.argv[1],int(sys.argv[2]))