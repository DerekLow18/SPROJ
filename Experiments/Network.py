import nest
import pylab
import nest.topology as topp
import nest.raster_plot as raster
import networkx as nx
import matplotlib.pyplot as plt
import numpy
import csv

#DEFINE FUNCTIONS
def drawNetwork(pop):
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
'''
def writeToCSV(adjMatrix):
	#writes the provided adjacency matrix to a CSV for viewing
	write_file = "~/Documents/SPROJ/Experiments/Syn Weights/syn_trial.csv"
	with open(write_file, "wb") as file:
		matrix_writer = csv.writer(file, delimiter=',')
	return
'''

def createNetwork(file_name, popSize):
	#generates an adj matrix based on parameters of network, and depending on formula
	#such as small world, clustering, criticality, etc. WIP
	adjMatrix = numpy.zeros((popSize, popSize))
	numpy.savetxt("./Syn Weights/"+file_name, adjMatrix, delimiter=",")
	return

def readAndConnect(file, population):
	'''Reads from a csv file for storing weights, connects corresponding
	nest neurons, outputs a numpy matrix'''
	matrix = numpy.loadtxt(open(file, "rb"), delimiter=",")
	row_pos = 0
	#adjMatrix = []
	for i_neuron_array in matrix:
		col_pos = 0
		for j_connection in i_neuron_array:
			if j_connection == 1.0:
				#adjMatrix.append([row_pos,col_pos])
				nest.Connect([population[row_pos]],[population[col_pos]])
			col_pos = col_pos + 1		
		row_pos = row_pos +1
	'''for i in adjMatrix:
		print "connection:",population[i[0]], " to ", population[i[1]]
		firstConnect = int(i[0])
		secondConnect = int(i[1])
		nest.Connect([population[firstConnect]], [population[secondConnect]])'''
	return matrix

def readAndCreate(file):
	'''Reads from a csv file for storing weights, creates the population indicated,
	returns population'''
	#read a csv file with conections, and rows and columns correspond to individual neurons
	matrix = numpy.loadtxt(open(file, "rb"), delimiter=",")
	#Set parameters of the network by reading the length of the matrix (number of arrays)
	numNeuronsCSV = len(matrix)
	numNeuronsInCSV = numpy.floor(numNeuronsCSV/5)
	numNeuronsExCSV = int(numNeuronsCSV-numNeuronsInCSV)

	#Create the neurons for the network
	pop = nest.Create("izhikevich", numNeuronsCSV)
	popEx = pop[:numNeuronsExCSV]
	popIn = pop[numNeuronsExCSV:]
	row_pos = 0
	#adjMatrix = []
	#Connect the neurons
	for i_neuron_array in matrix:
		col_pos = 0
		for j_connection in i_neuron_array:
			if j_connection == 1.0:
				#adjMatrix.append([row_pos,col_pos])
				nest.Connect([pop[row_pos]],[pop[col_pos]])
			col_pos = col_pos + 1		
		row_pos = row_pos +1
	return pop, popEx, popIn

'''def connectWithWeights(adjmatrix, pop):
	for i in pop:
		for j in pop:
			syn_weight = adjmatrix [i-1][j-1]
			syn_dict = {"weight": syn_weight}
			nest.Connect([i],[j],syn_spec = syn_dict)
	return'''

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
-get the model to read from an adj matrix csv
-write a separate coding segment that will generate a csv with specific network parameters such as small-worldness, average links, etc
-downsample

'''
######################################################################################
'''
#SET PARAMETERS
numNeurons = 10
numNeuronsIn = numpy.floor(numNeurons/5)
numNeuronsEx = int(numNeurons-numNeuronsIn)

#Create the neurons for the network
pop = nest.Create("izhikevich", numNeurons)
popEx = pop[:numNeuronsEx]
popIn = pop[numNeuronsEx:]
'''
createNetwork("foo.csv",10)
neuronPop, neuronEx, neuronIn = readAndCreate("./Syn Weights/syn_weights1.csv")

#CREATE NODES
noiseEx = nest.Create("poisson_generator",len(neuronEx),{'rate':1000.00})
noiseIn = nest.Create("poisson_generator", len(neuronIn),{'rate':100.00})
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
'''nest.Connect(popEx, pop, conn_dict, syn_spec = syn_dict_ex)
nest.Connect(popIn, pop, conn_dict, syn_spec = syn_dict_in)'''
nest.Connect(noiseEx, neuronEx)
nest.Connect(noiseIn, neuronIn)
nest.Connect(neuronEx, spikesEx)
nest.Connect(neuronIn, spikesIn)

#nest.Connect(multimeter, [1])
#nest.Connect(sine, [1])
#nest.Connect([pop[1]],[pop[2]])
#readAndConnect("./Syn Weights/syn_weights1.csv",pop)

#show me the connections
#print(nest.GetConnections())
#nest.PrintNetwork()
#print nest.GetConnections(pop1)
#print(makeAdjMatrix(popEx))

nest.Simulate(1000.0)

pylab.figure(2)
drawNetwork(neuronPop)
plot = nest.raster_plot.from_device(spikesEx, hist=True)

'''
The exact neuron spikes and corresponding timings can be obtained by viewing the events
dictionary of GetStatus(spikesEx, "events")
'''
print nest.GetStatus(spikesEx, "events")
plt.show()
