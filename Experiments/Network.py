import nest
import pylab
import nest.topology as topp
import nest.raster_plot as raster
import networkx as nx
import matplotlib.pyplot as plt
import numpy

#DEFINE FUNCTIONS
def genNetwork(pop):
	"""take a nest network and generate a networkX graph"""
	pop_connect_dict = nest.GetConnections(pop)
	G = nx.DiGraph()
	for i in pop:
		G.add_node(i)
	netXEdges = []
	for j in pop_connect_dict:
		x = j[0]
		y = j[1]
		netXEdges.append((x,y))
	G.add_edges_from(netXEdges)
	nx.draw(G, with_labels=True)
	return G

def readAndConnect(file):
	"""Reads from a csv file for storing weights, connects corresponding
	nest neurons, outputs a numpy matrix"""
	matrix = numpy.loadtxt(open(file, "rb"), delimiter=",")
	'''in_pos = 
	for i in matrix:
		print i
		for j in i:
			if i[j] == 1:
				nest.Connect([i],[j])'''
	return matrix

'''def connectWithWeights(adjmatrix, pop):
	for i in pop:
		for j in pop:
			syn_weight = adjmatrix [i-1][j-1]
			syn_dict = {"weight": syn_weight}
			nest.Connect([i],[j],syn_spec = syn_dict)
	return'''

def makeAdjMatrix(pop):
	"""take a nest network and generate an adjacency matrix,
	where 1 = connected, 0 = not connected. Reads from a csv file for storing weights"""
	pop_connect_dict = nest.GetConnections(pop)
	numPop = len(pop)
	pcdMatrix = numpy.zeros((numPop,numPop))
	for i in pop_connect_dict:
		src=i[0]
		dest=i[1]
		pcdMatrix[src-1][dest-1] = 1
	return pcdMatrix

def rasterGenerator(pop):
	spikes = nest.Create("spike_detector",len(pop))
	nest.Connect(pop, spikes)
	plot = nest.raster_plot.from_device(spikes, hist=True)
	return plot

######################################################################################

#     #    #####    #####     ######   #
##   ##   #     #   #    #    #        #
# # # #   #     #   #     #   #        #
#  #  #   #     #   #     #   ######   #
#     #   #     #   #     #   #        #
#     #   #     #   #    #    #        #
#     #    #####    #####     ######   ######

#########################################
'''
TO DO:
Control so inhibitory neurons are not connected to inhibitory neurons.



'''
######################################################################################

#SET PARAMETERS
numNeurons = 50
numNeuronsIn = numpy.floor(numNeurons/5)
numNeuronsEx = int(numNeurons-numNeuronsIn)

#CREATE NODES
pop = nest.Create("izhikevich", numNeurons)
popEx = pop[:numNeuronsEx]
popIn = pop[numNeuronsEx:]
noiseEx = nest.Create("poisson_generator",len(popEx),{'rate':300.00})
noiseIn = nest.Create("poisson_generator", len(popIn),{'rate':100.00})
#sine = nest.Create("ac_generator",1,{"amplitude": 100.0, "frequency" :2.0})
spikes = nest.Create("spike_detector", 2)
spikesEx = spikes[:1]
spikesIn = spikes[1:]

#nest.SetStatus(pop, {"I_e": 376.0})
#multimeter to detect membrance potential
#multimeter = nest.Create("multimeter")
#nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})

Ex = 1
d = 1.0
wEx = .01
wIn = -.05

#SPECIFY CONNECTION DICTIONARIES
conn_dict = {"rule": "fixed_indegree", "indegree": Ex,
			"autapses":False,"multapses":False} #connection dictionary
syn_dict_ex = {"delay": d, "weight": wEx}
syn_dict_in = {"delay": d, "weight": wIn}

#SPECIFY CONNECTIONS
#nest.Connect(pop1, pop1, conn_dict, syn_spec = syn_dict_ex)
nest.Connect(popEx, pop, conn_dict, syn_spec = syn_dict_ex)
nest.Connect(popIn, pop, conn_dict, syn_spec = syn_dict_in)
nest.Connect(noiseEx, popEx)
nest.Connect(noiseIn, popIn)
#nest.Connect(sine, [1])
nest.Connect(popEx, spikesEx)
nest.Connect(popIn, spikesIn)
#nest.Connect(multimeter, [1])

#show me the connections
#print(nest.GetConnections())
#nest.PrintNetwork()
#print nest.GetConnections(pop1)
#print(makeAdjMatrix(popEx))

nest.Simulate(1000.0)

#pylab.figure(2)
#genNetwork(pop1)
plot = nest.raster_plot.from_device(spikesEx, hist=True)
plt.show()

#print(readAndConnect("./Syn Weights/syn_weights1.csv"))
