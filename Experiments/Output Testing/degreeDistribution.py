import numpy as np
import os, sys

#calculate the in and outgoing number of connections for each neuron
def calculateInAndOut(adjMatrix):
	matrix = np.genfromtxt(adjMatrix,delimiter = ',')
	indegrees = np.sum(matrix,axis = 1)	
	outdegrees = np.sum(matrix,axis=0)
	totaldegrees = indegrees + outdegrees
	return indegrees, outdegrees, totaldegrees

#calculate the number of neurons with a particular number of incoming and outgoing connections
#then produce the indegree distributions and outdegree distributions
def calculateDD(indegrees, outdegrees, totaldegrees):
	#store the different counts
	indegreesCounts = np.zeros(max(int(indegrees.max()),int(outdegrees.max()))+1)
	outdegreesCounts = np.zeros(len(indegrees))
	totalDegreeCounts = np.zeros(int(totaldegrees.max())+1)
	for i in indegrees:
		indegreesCounts[int(i)] += 1
	for j in outdegrees:
		outdegreesCounts[int(j)] += 1
	for k in totaldegrees:
		totalDegreeCounts[int(k)] +=1 
	#divide by length because length is equivalent to the population size
	return indegreesCounts/len(indegrees), outdegreesCounts/len(indegrees), totalDegreeCounts/len(indegrees)

if __name__=='__main__':
	if len(sys.argv) < 2:
		raise Exception("Need a weight matrix in csv format to analyze!")
		exit()
	incoming, outgoing, total = calculateInAndOut(sys.argv[1])
	print(incoming, outgoing,total)
	incomingDD, outgoingDD, totalDD = calculateDD(incoming,outgoing,total)
	print("Icoming DD:",incomingDD,"\n Outgoing DD:",outgoingDD,"\n totalDD:",totalDD)
