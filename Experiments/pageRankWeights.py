'''
Author: Derek Low

This algorithm will take a csv of weights and produce a new csv that uses page ranks to update weights.
It will iterate and make updates to the pageRanks until the values converge (i.e. the pageRank from an iteration is within
a certain margin of the page rank from the previous iteration)

'''
import nest
import pylab
import networkx as nx
import matplotlib.pyplot as plt
import numpy

class pageRankOnWeights(object):

	def __init__(self, csv):
		self.csv = numpy.loadtxt(open(csv, "rb"), delimiter=",")
		self.rankDict = {}
		self.initializeRankDict()

	def initializeRankDict(self):
		'''
		initialize the rank dictionary of the neurons as simply the number of incoming connections.

		Since pageRank never changes if all-to-all connected and all initial page ranks are 1, two options:
		1. Initialize page rank withvalues other than 1
		2. Prune connections
		
		Additionally, initialize the dictionary so that each neuron has two entries: pageRank, and number of outgoing connections

		'''
		#this first implementation merely sets all pageRanks to 1 for all neurons, and depends on arbitrarily pruned network described in the below createRandomNetwork method
		for i in range(len(self.csv)):
			self.rankDict[i] = {}
			self.rankDict[i]['pageRank'] = 1
			self.rankDict[i]['num_outgoing_connections'] = 0
		#Set the number of outgoing connections, where row = source, column = target in csv. Any value over 0 counts as an outgoing connection in this case
		for ith_neuron_index in range(len(self.csv)):
			for jth_neuron_index in range(len(self.csv)):
				if self.csv[ith_neuron_index][jth_neuron_index] != 0:
					self.rankDict[ith_neuron_index]['num_outgoing_connections'] += 1

		#TODO: this second implemntation will initalize page rank based on some arbitrary value, perhaps determined by initial weights
		return

	def assignPageRank(self):
		'''
		Assign page rank to node based on number of incoming and outgoing connections
		'''
		
		for rank in self.rankDict:
			self.rank
		
		return

	def updateWeights(self):
		'''
		Rules for weight updating are stored in this function
		'''
		return

	def printStuff(self):
		'''
		print shit in here
		'''
		print self.rankDict
		return


'''
Paramaterize for creating a random network. Optionally prune connections, depending on how you want to
initialize the pageRank for each neuron

'''
weightThreshold = 0.3 #to prune connections, set a weight threshold and weights below will be set to 0

def createRandomNetwork(file_name, popSize):
	'''
	generates an adj matrix based on parameters of network, and depending on formula
	such as small world, clustering, criticality, etc. WIP
	'''
	adjMatrix = numpy.zeros((popSize, popSize))
	#ensure that there are no self-connections at each neuron
	for ith_neuron_index in range(len(adjMatrix)):
		adjMatrix[ith_neuron_index][ith_neuron_index] = 0

	#fill out the adj matrix for all other neurons
	for ith_neuron_index in range(len(adjMatrix)):
		for jth_neuron_index in range(len(adjMatrix)):
			if ith_neuron_index != jth_neuron_index:
				adjMatrix[ith_neuron_index][jth_neuron_index]= numpy.random.ranf()
				if adjMatrix[ith_neuron_index][jth_neuron_index] <= weightThreshold:
					adjMatrix[ith_neuron_index][jth_neuron_index] = 0
	'''
	Since pageRank never changes if all-to-all connected and all initial page ranks are 1, two options:
	1. Initialize page rank withvalues other than 1
	2. Prune connections

	Since we are already generating the random matrix, let's arbitrarily threshold to prune the certain connections
	'''
	numpy.savetxt(file_name, adjMatrix, delimiter=",")
	return	


def main():
	createRandomNetwork("initPRWeights.csv",5)
	pageRanks = pageRankOnWeights("initPRWeights.csv")

	pageRanks.printStuff()

if __name__ =="__main__":
	main()