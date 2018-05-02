#Modularity: Maximizes the modularity measure and gives you the corresponding modules.
import networkx as nx
import modularity_maximization as mm
import numpy as np
import sys

def createRandomNetwork(file_name, popSize):
	#generates an adj matrix based on parameters of network, and depending on formula
	#such as small world, clustering, criticality, etc. WIP
	adjMatrix = np.zeros((popSize, popSize))
	#ensure that there are no self-connections at each neuron
	for ith_neuron_index in range(len(adjMatrix)):
		adjMatrix[ith_neuron_index][ith_neuron_index] = 0

	#fill out the adj matrix for all other neurons
	for ith_neuron_index in range(len(adjMatrix)):
		for jth_neuron_index in range(len(adjMatrix)):
			if ith_neuron_index != jth_neuron_index:
				adjMatrix[ith_neuron_index][jth_neuron_index]= np.random.randint(0,high=2)
	print adjMatrix
	np.savetxt(file_name, adjMatrix, delimiter=",")
	return	

def matrixToNxGraph(filename):
	adjMatrix = np.genfromtxt(filename, delimiter = ',')
	rows, cols = np.where(adjMatrix == 1)
	edges = zip(rows.tolist(),cols.tolist())
	graph = nx.Graph()
	graph.add_edges_from(edges)
	return graph

def main():
	if sys.argv < 2:
		print("Usage: python modularity.py name_of_my_adjMatrix.csv")
		exit()
	createRandomNetwork("network.csv",20)
	graph = matrixToNxGraph(sys.argv[1])
	comm_dict = mm.partition(graph)
	print(comm_dict)
	for comm in set(comm_dict.values()):
	    print("Community %d"%comm)
	    print(', '.join([node for node in comm_dict if comm_dict[node] == str(comm)]))
	print('Modularity of such partition for graph is %.3f' %mm.get_modularity(graph, comm_dict))

main()