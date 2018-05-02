'''
Author: Derek Low

This algorithm will take a csv of weights and produce a new csv that uses page ranks to update weights.
It will iterate and make updates to the pageRanks until the values converge (i.e. the pageRank from an iteration is within
a certain margin of the page rank from the previous iteration)

'''
import pylab
import matplotlib.pyplot as plt
import numpy
import copy
import sys
import argparse
import networkx as nx

class pageRankOnWeights(object):

	#def __init__(self, csv = "./Syn Weights/randomWeights1.csv"):
	def __init__(self,csv):
		self.connections = numpy.loadtxt(open(csv, "rb"), delimiter=",")
		self.prunedConnections = []
		self.maxIterations = 1000
		#Store the transpose of the neuron, which will be helpful in calculating the weight
		self.incomingWeight = (copy.deepcopy(self.connections)).transpose()
		self.rankDict = {}
		self.dampingFactor = 0.8
		#self.normFactor = 2.2
		self.finalMatrix = []
		self.initializeRankDict()
		self.pruneOriginal()

	def initializeRankDict(self):
		'''
		initialize the rank dictionary of the neurons as simply the number of incoming connections.

		Since pageRank never changes if all-to-all connected and all initial page ranks are 1, two options:
		1. Initialize page rank withvalues other than 1
		2. Prune connections
		
		Additionally, initialize the dictionary so that each neuron has two entries: pageRank, and number of outgoing connections

		'''
		#this first implementation merely sets all pageRanks to 1 for all neurons, and depends on arbitrarily pruned network described in the below createRandomNetwork method
		for i in range(len(self.connections)):
			self.rankDict[i] = {}
			self.rankDict[i]['pageRank'] = 1
		
		#The code below is for setting # of outgoing connections for each neuron, but is not necessary with the updated formula
			self.rankDict[i]['num_outgoing_connections'] = 0
		#Set the number of outgoing connections, where row = source, column = target in csv. Any value over 0 counts as an outgoing connection in this case
		for ith_neuron_index in range(len(self.connections)):
			for jth_neuron_index in range(len(self.connections)):
				if self.connections[ith_neuron_index][jth_neuron_index] != 0:
					self.rankDict[ith_neuron_index]['num_outgoing_connections'] += 1
		

		#TODO: this second implemntation will initalize page rank based on some arbitrary value, perhaps determined by initial weights
		return

	def pruneOriginal(self):
		connectionMatrix = self.connections
		outputMatrix = numpy.zeros((len(connectionMatrix), len(connectionMatrix)))
		for sourceIndex in range(len(connectionMatrix)):
			for targetIndex in range(len(connectionMatrix)):
				if connectionMatrix[sourceIndex][targetIndex] >= connectionMatrix[targetIndex][sourceIndex]:
					outputMatrix[sourceIndex][targetIndex] = 1
					outputMatrix[targetIndex][sourceIndex] = 0
				elif connectionMatrix[sourceIndex][targetIndex] < connectionMatrix[targetIndex][sourceIndex]:
					outputMatrix[sourceIndex][targetIndex] = 0
					outputMatrix[targetIndex][sourceIndex] = 1
		self.prunedConnections = outputMatrix
		return

	def assignPageRank(self):
		'''
		Assign page rank to node. This will be done via neurons to incoming connections, as well as weights

		The basic formulation of pageRank is as follows:

		PR(a) = 1-d + d(sum(PR(n)/N(n)) for all neurons n incoming to a)

		where:
		d is damping factor
		a is the current neuron you are finding the page rank for
		N = number of outgoing connections from neuron n (share the source pageRank equally to all targets)
		
		However, since we have an all-to-all network, with weights, we can change the formulation
		of page rank a little bit to account for the weights. Additionally, the neurons do not
		'share' a pageRank equally. Instead of dividing by number of connections, we will multiply
		by the weight of the connection.

		PR(a) = (1-d) + d(sum(PR(n)*W(n -> a)) for all n with target a)

		where:
		w is the weight of the connection flowing from neuron n to target a, or our current neuron

		'''
		notConverged = True
		numIterations = 0
		#run this until the values converge
		while notConverged:
			priorRankDict = copy.deepcopy(self.rankDict)
			#print priorRankDict
			for neuron in range(len(self.rankDict)):
				runningSum = 0
				for source in range(len(self.incomingWeight[neuron])):
					#sum the pageRank of incoming connections, multiply by the weights of those connections
					if self.incomingWeight[neuron][source] == 0:
						runningSum += 0
					else:
						#runningSum += priorRankDict[source]['pageRank']/float(self.rankDict[source]['num_outgoing_connections'])
						#print "prior page rank is " + str(priorRankDict[source]['pageRank'])
						#changing the denominator of each summation factor kind of changes the results?
						runningSum += (self.rankDict[source]['pageRank']*self.incomingWeight[neuron][source])/float(numpy.sum(self.connections[source]))
				#print "running sum is " + str(runningSum)
				#this is the standard method, with damping factor
				self.rankDict[neuron]['pageRank'] = (1-self.dampingFactor) + self.dampingFactor*runningSum
				#lets do this in a normalization factor way:
				#self.rankDict[neuron]['pageRank'] = self.normFactor*runningSum

				
			numIterations += 1
			#check for convergence, or if maxIterations have been reached
			if priorRankDict == self.rankDict or numIterations == self.maxIterations:
				notConverged = False
		print "Convergence took %s iterations. The pageRanks are:" %numIterations
		print self.rankDict
		return

	def updateWeights(self):
		'''
		Rules for weight updating are stored in this function
		'''
		newConnections = copy.deepcopy(self.connections)
		for sourceIndex in range(len(self.connections)):
			for targetIndex in range(len(self.connections)):
				newConnections[sourceIndex][targetIndex] *= self.rankDict[sourceIndex]['pageRank']
		print self.connections
		print ('The pageRank-updated connection matrix is:')
		print newConnections
		return newConnections

	def pruneNetwork(self):
		'''
		Take the updated weight matrix and use the new weight values to prune connections based on the weights
		Higher weights stay, lower weights go
		'''
		connectionMatrix = self.updateWeights()
		outputMatrix = numpy.zeros((len(connectionMatrix), len(connectionMatrix)))
		for sourceIndex in range(len(connectionMatrix)):
			for targetIndex in range(len(connectionMatrix)):
				if connectionMatrix[sourceIndex][targetIndex] >= connectionMatrix[targetIndex][sourceIndex]:
					outputMatrix[sourceIndex][targetIndex] = 1
					outputMatrix[targetIndex][sourceIndex] = 0
				elif connectionMatrix[sourceIndex][targetIndex] < connectionMatrix[targetIndex][sourceIndex]:
					outputMatrix[sourceIndex][targetIndex] = 0
					outputMatrix[targetIndex][sourceIndex] = 1

		self.finalMatrix = outputMatrix
		numpy.savetxt('./Syn Weights/pageRankAdjConn.csv', outputMatrix, delimiter=",")
		return outputMatrix

	def printStuff(self):
		'''
		print shit in here
		'''
		print self.rankDict
		#print self.connections
		#print self.incomingWeight
		return

#How much has the matrix changed between the input matrix and the pageRank modified matrix?
def checkChange(original, modified):
	connectionMatrix = original
	outputMatrix = numpy.zeros((len(connectionMatrix), len(connectionMatrix)))
	for sourceIndex in range(len(connectionMatrix)):
		for targetIndex in range(len(connectionMatrix)):
			if connectionMatrix[sourceIndex][targetIndex] >= connectionMatrix[targetIndex][sourceIndex]:
				outputMatrix[sourceIndex][targetIndex] = 1
				outputMatrix[targetIndex][sourceIndex] = 0
			elif connectionMatrix[sourceIndex][targetIndex] < connectionMatrix[targetIndex][sourceIndex]:
				outputMatrix[sourceIndex][targetIndex] = 0
				outputMatrix[targetIndex][sourceIndex] = 1
	squashOriginal = outputMatrix.flatten()
	squashMod = modified.flatten()
	changed = numpy.sum(squashOriginal != squashMod)
	print(squashOriginal)
	print(squashMod)
	print(str(changed) + " connections have switched.")
	return squashOriginal, squashMod

#display the prior and recreated populations
def drawNetwork(pop):
	G = nx.DiGraph()
	for i in range(len(pop)):
		G.add_node(i)
	netXEdges = []
	for i in range(len(pop)):
		for j in range(len(pop)):
			if pop[i][j] == 1:
				netXEdges.append((i,j))
	G.add_edges_from(netXEdges)
	nx.draw(G, with_labels=True)
	return G


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
	numpy.savetxt(file_name, adjMatrix, delimiter=",")
	return	


def main():
	parser = argparse.ArgumentParser(description = "Use pageRank to modify the network.")
	parser.add_argument('-csv', action="store", dest = 'csv',default = "./Syn Weights/randomWeights1.csv")
	parser.add_argument('-iters', action="store", dest = 
		'iters',type=int, default = 1)
	inputs = parser.parse_args()
	'''
	if len(sys.argv) == 3:
		pageRanks = pageRankOnWeights(sys.argv[1])
		numIterations = sys.argv[2]
	if len(sys.argv) != 3:
		print("If you don't enter as: %s weight.csv iterations, random weight csv will be generated with 10 pop size, and 1 iteration." % sys.argv[0])
		createRandomNetwork("./Syn Weights/randomWeights.csv",10)
		pageRanks = pageRankOnWeights()
		numIterations = 1
	
	if inputs.csv == "./Syn Weights/randomWeights.csv":
		createRandomNetwork(inputs.csv,10)
		pageRanks = pageRankOnWeights()
	else:
		'''
	pageRanks = pageRankOnWeights(inputs.csv)
	pageRanks.assignPageRank()
	pageRanks.pruneNetwork()
	for i in range(inputs.iters):
		pageRanks1 = pageRankOnWeights('./Syn Weights/pageRankAdjConn.csv')
		pageRanks1.assignPageRank()
		pageRanks1.pruneNetwork()
	print("The original is",pageRanks.prunedConnections)
	print("The final is", pageRanks1.finalMatrix)
	checkChange(pageRanks.connections, pageRanks1.finalMatrix)

	#origAdjMat = nx.from_numpy_matrix(prunedOrigin, create_using = nx.Digraph())
	#print(pageRanks1.finalMatrix)
	#endAdjMat = nx.from_numpy_matrix(pageRanks1.finalMatrix, create_using= nx.DiGraph())
	#nx.draw(endAdjMat)
	drawNetwork(pageRanks.prunedConnections)
	plt.show()
	drawNetwork(pageRanks1.finalMatrix)
	plt.show()


if __name__ =="__main__":
	main()