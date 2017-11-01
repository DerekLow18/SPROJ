#Derek Low
#Scipts for generating adjacency matrices for graphs with interesting features

import numpy
import networkx as nx

class DirectedAdj(object):

	def __init__(self, size):
		self.size = size #size of network
		self.adjMatrix = numpy.zeros((self.size, self.size))

	def createAdjMatrix(self, graph):

		#ensure there are no self-connections for any neurons
		for i_neuron_array in range(len(adjMatrix)):
			self.adjMatrix[i_neuron_array][i_neuron_array] = 0

		#fill out the array for all other neurons
		for ith_neuron_index in range(len(adjMatrix)):
			for jth_neuron_index in range(len(adjMatrix)):
				if ith_neuron_index != jth_neuron_index:
					self.adjMatrix[i_neuron_array][j_connection_value]= numpy.random.randint(0,high=2)
		return	


	def write(self, file_name):
		numpy.savetxt("./Syn Weights/"+file_name, self.adjMatrix, delimiter=",")
		return

class RandomDirectedAdj(DirectedAdj):

	def createRandAdjMatrix(self):

		#ensure there are no self-connections for any neurons
		for i_neuron_array in range(len(self.adjMatrix)):
			self.adjMatrix[i_neuron_array][i_neuron_array] = 0

		#fill out the array for all other neurons
		for ith_neuron_index in range(len(self.adjMatrix)):
			for jth_neuron_index in range(len(self.adjMatrix)):
				if ith_neuron_index != jth_neuron_index:
					self.adjMatrix[i_neuron_array][j_connection_value]= numpy.random.randint(0,high=2)
		return	



class ScaleFreeAdj(DirectedAdj):

	def __init__(self, size, degree_distribution):
		self.ddist = degree_distribution #create a directed graph

	def createDDMatrix(self):
		G = nx.scale_free_graph(self.size, create_using = DiGraph)
		G.remove_edges_from(G.selfloop_edges())
		
