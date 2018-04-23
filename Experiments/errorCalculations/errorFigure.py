import sys, os, fnmatch, re
import numpy as np
import matplotlib.pyplot as plt
import re

def produceErrorFigure(fileName, errorArrayInd):
	errorArray = np.genfromtxt(str(fileName),delimiter = ',').transpose()
	fileID = os.path.splitext(str(fileName))[0]
	print(errorArray)
	plt.scatter(range(len(errorArray[int(errorArrayInd)])),errorArray[int(errorArrayInd)],s=5)
	plt.xlabel("Iteration")
	plt.ylabel("Mean Squared Error")
	plt.savefig("../../Main Writing/Figures/Errors/"+fileID + "Error.svg",format = 'svg')
	plt.show()

#produce a scatter plot from a csv file, depicting error changes
if len(sys.argv) < 3:
	raise Exception("Please provide a csv file of the error, and the array the error is in.")
	exit()
produceErrorFigure(sys.argv[1],sys.argv[2])
