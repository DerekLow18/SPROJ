import numpy as np
import os, sys, fnmatch
import matplotlib.pyplot as plt
from itertools import zip_longest
from matplotlib.ticker import MaxNLocator
from scipy import stats
'''
Across all samples of the same origin, take the resulting matrix at
threhsold = 0.50 and calculate the degree distribution. Take the average
degree distribution, and plot it.
'''


#calculate the in and outgoing number of connections for each neuron
def calculateInAndOut(adjMatrix):
	matrix = np.genfromtxt(adjMatrix,delimiter = ',')
	#sum the matrix by its columns
	indegrees = np.sum(matrix,axis = 1)
	#sum the matrix by its rows
	outdegrees = np.sum(matrix,axis=0)
	#add the two resulting arrays for total number of connections for each neuron
	totaldegrees = indegrees + outdegrees
	return indegrees, outdegrees, totaldegrees

#calculate the number of neurons with a particular number of incoming and outgoing connections
#then produce the indegree distributions and outdegree distributions
def calculateDD(indegrees, outdegrees, totaldegrees):
	#store the different counts
	indegreesCounts = np.zeros(max(int(indegrees.max()),int(outdegrees.max()))+1)
	outdegreesCounts = np.zeros(len(indegreesCounts))
	totalDegreeCounts = np.zeros(int(totaldegrees.max())+1)
	for i in indegrees:
		indegreesCounts[int(i)] += 1
	for j in outdegrees:
		outdegreesCounts[int(j)] += 1
	for k in totaldegrees:
		totalDegreeCounts[int(k)] +=1 
	#divide by length because length is equivalent to the population size
	return indegreesCounts/len(indegrees), outdegreesCounts/len(indegrees), totalDegreeCounts/len(indegrees)

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

if __name__=='__main__':
	if len(sys.argv) < 2:
		raise Exception("Need a weight matrix in csv format to analyze!")
		exit()
	numFiles = 0
	sumIncomingDD = []
	sumOutgoingDD = []
	sumTotalDD = []
	sumStdDev = 0
	path=str(sys.argv[1])
	for subdir in os.listdir(path):
		if fnmatch.fnmatch(subdir,"*downsample"):
			for file in os.listdir(path+subdir):
				if fnmatch.fnmatch(file,"50weightMatrix.csv"):
					numFiles += 1
					#calculate the number of connections along each axis, as well as the total
					#number of connections
					incoming, outgoing, total = calculateInAndOut(path+subdir+"/"+file)
					#calculate the number of neurons with a particular degree
					incomingDD, outgoingDD, totalDD = calculateDD(incoming,outgoing,total)
					sumIncomingDD=[x+y for x,y in zip_longest(sumIncomingDD, incomingDD, fillvalue=0)]
					sumOutgoingDD=[x+y for x,y in zip_longest(sumOutgoingDD, outgoingDD, fillvalue=0)]
					sumTotalDD=[x+y for x,y in zip_longest(sumTotalDD, totalDD, fillvalue=0)]
					sumStdDev = sumStdDev + np.std(sumTotalDD)
					print(np.std(sumTotalDD))
					#print("Icoming DD:",incomingDD,"\n Outgoing DD:",outgoingDD,"\n totalDD:",totalDD)
	
	avgIncomingDD, avgOutgoingDD, avgTotalDD = np.array(sumIncomingDD)/numFiles,np.array(sumOutgoingDD)/numFiles,np.array(sumTotalDD)/numFiles
	#print(avgIncomingDD,avgOutgoingDD,avgTotalDD)
	plotIncomingDD = zero_to_nan(avgIncomingDD)
	plotOutgoingDD = zero_to_nan(avgOutgoingDD)
	plotTotalDD = zero_to_nan(avgTotalDD)
	print("Results from number of files:",numFiles)
	print(plotTotalDD)
	print("standard error for total DD:", sumStdDev/numFiles)
	print("max is:", max(plotTotalDD))
	print("min is",min(plotTotalDD))
	print("ks test:",stats.shapiro(avgTotalDD[45:96]))
	fig = plt.figure()
	ax0=fig.add_subplot(111)
	res = stats.probplot(avgTotalDD[45:96], dist="norm", plot=plt)
	ax0.get_lines()[0].set_marker('o')
	ax0.get_lines()[0].set_markersize(5.0)
	ax0.get_lines()[1].set_linewidth(2.0)
	ax0.get_lines()[1].set_color('c')
	#plt.savefig("../../Main Writing/Figures/DD/simulatedvarQQpop50.svg",format = 'svg')
	plt.show()
	print(res[1])
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	plt.scatter(range(len(avgTotalDD)),plotTotalDD,s=5)
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.ylabel("Pr[D=k]")
	plt.xlabel("Number of Connections")
	#plt.ylim(0,0.16)
	#plt.yscale('log')
	#plt.xscale('log')
	#plt.savefig("../../Main Writing/Figures/DD/simulatedvarDDPop10.svg",format = 'svg')
	plt.show()