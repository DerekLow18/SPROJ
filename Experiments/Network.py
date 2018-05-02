import pylab
import networkx as nx
import matplotlib.pyplot as plt
import numpy
import time
import nest.raster_plot as raster
import nest.topology as topp
import nest
import sys

#DEFINE FUNCTIONS
'''
Take a NEST network as a parameter and generates a networkX graph
'''
#suppress terminal print updates from nest functions
nest.set_verbosity("M_WARNING")

def drawNetwork(pop):
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
Create a random network (random being defined by the numpy.random.randint function_)
and write it to a csv. Honestly, easier way to do this is numpy.random.randint(0,high=2,10,10)
or something like that, as numpy will create a 2d array with those randints.

However, I did it this way as an exploration into how I could manipulate these arrays for
future purposes, such as implementing other network properties.
'''
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
	numpy.savetxt("./Syn Weights/"+file_name, adjMatrix, delimiter=",")
	return	

'''
Reads from a csv file for storing weights, connects corresponding
nest neurons, outputs a numpy matrix
'''
def readAndConnect(file, population):
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

	pop = nest.Create("izhikevich", len(matrix))
	ratio = .2 # inhib to excite
	#Connect the neurons
	for row_pos in range(len(matrix)):
		for col_pos in range(len(matrix[row_pos])):
			if matrix[row_pos][col_pos] == 1.0:
				nest.Connect([pop[row_pos]],[pop[col_pos]],syn_spec = {"model":"stdp_synapse","weight":(-1.0 if numpy.random.random() <= ratio else 1.0)})
	#Set parameters of the network by reading the length of the matrix (number of arrays)
	'''numNeuronsCSV = len(matrix)
	numNeuronsInCSV = numpy.floor(numNeuronsCSV/5)
	numNeuronsExCSV = int(numNeuronsCSV-numNeuronsInCSV)

	#Create the neurons for the network
	pop = nest.Create("iaf_psc_alpha_presc", numNeuronsCSV)
	#the first 1/5 neurons are inhibitory, the rest are excitatory
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
				#nest.Connect([pop[row_pos]],[pop[col_pos]])
				
				if row_pos <= numNeuronsInCSV:
					#print "inhib connected"
					nest.Connect([pop[row_pos]],[pop[col_pos]],syn_spec = {"model":"stdp_synapse","weight":-50.0})
				else:
					nest.Connect([pop[row_pos]],[pop[col_pos]],syn_spec={"model":"stdp_synapse","weight":50.0})
			col_pos = col_pos + 1		
		row_pos = row_pos +1'''
	return pop, matrix

def spikeTimeMatrix(spikes, numNeurons, timesteps):
	# takes saved matrix where first row is spiking neuron
	# and second row is time of spike
	# note: loses precision
	output = numpy.matrix(numpy.zeros((numNeurons, timesteps)))
	for i in range(len(spikes[0])):
		output[int(spikes[0][i]-1), int(round(spikes[1][i]))] = 1
	return output

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
-write a separate coding segment that will generate a csv with specific network parameters such as small-worldness, average links, etc
-downsample
'''
######################################################################################
def main(num):
	nest.ResetKernel()
	msd = int(time.time())
	N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
	nest.SetKernelStatus({'rng_seeds': range(msd+N_vp+1,msd+2*N_vp+1)})

	#SET PARAMETERS
	#numNeurons = 50
	#cE = float((.8*numNeurons)/10)
	poisson_rate = 50.0 #1000.0*((2.0*30.0)/(0.1*20.0*cE))*cE
	#createRandomNetwork("groundTruth50.csv",10)
	neuronPop, popMatrix = readAndCreate("./Syn Weights/sfNetworkPop50.csv")
	#neuronPop, popMatrix = readAndCreate("./Syn Weights/groundTruth50.csv")

	#CREATE NODES
	noise = nest.Create("poisson_generator",1,{'rate':poisson_rate})
	#noiseIn = nest.Create("poisson_generator",1,{'rate':10000.0})
	#sine = nest.Create("ac_generator",1,{"amplitude": 100.0, "frequency" :2.0})
	spikes = nest.Create("spike_detector", 1)
	#spikesEx = spikes[:1]
	#spikesIn = spikes[1:]

	Ex = 1
	d = 20.0
	wEx = 10.0
	wIn = -5.0

	#SPECIFY CONNECTION DICTIONARIES
	conn_dict = {"rule": "fixed_indegree", "indegree": Ex,
				"autapses":False,"multapses":False} #connection dictionary
	syn_dict_ex = {"delay": d, "weight": wEx}
	syn_dict_in = {"delay": d, "weight": wIn}

	#SPECIFY CONNECTIONS
	'''
	for i in range(len(neuronPop)):
		if numpy.random.random() <= .4:
			nest.Connect(noise,[i],syn_spec = syn_dict_ex)
	'''
	nest.Connect(noise,neuronPop,syn_spec=syn_dict_ex)
	nest.Connect(neuronPop,spikes)

	#readAndConnect("./Syn Weights/syn_weights1.csv",pop)
	simTime = 10000.0
	nest.Simulate(simTime)
	n = nest.GetStatus(spikes, "events")[0]
	temp = numpy.array([n['senders'], n['times']])
	fullMatrix = spikeTimeMatrix(temp, len(neuronPop), int(simTime))
	numpy.savetxt("./Spike Results/pop50sf/%02didTimes.csv" % (num),fullMatrix,delimiter=',')
	numpy.savetxt("./Spike Results/pop50sf/%02dspikeTrains.csv" % (num),temp,delimiter = ',')
	#pylab.figure(2)
	plot = nest.raster_plot.from_device(spikes, hist=True)
	#for i in range(len(neuronPop)):
	#	nest.DisconnectOneToOne([i], [52],syn_dict_ex)
	#drawNetwork(neuronPop)
	'''
	The exact neuron spikes and corresponding timings can be obtained by viewing the events
	dictionary of GetStatus(spikesEx, "events")
	'''
	#print nest.GetStatus(spikes, "events")
	#print nest.GetStatus(nest.GetConnections(neuronPop, synapse_model = 'stdp_synapse'))
	plt.show()
if __name__=="__main__":
	if len(sys.argv) < 2:
		print("Incorrect number of arguments. Please state number of iterations")
		exit()

	numGraphs = int(sys.argv[1])
	initIdx = 1
	for i in range(initIdx, initIdx + numGraphs):
		sys.stdout.write("\r" +"Simulation number: " + str(i))
		sys.stdout.flush()
		main(i)
