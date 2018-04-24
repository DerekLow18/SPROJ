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
'''
if len(sys.argv) < 3:
	raise Exception("Please provide a csv file of the error, and the array the error is in.")
	exit()
'''
#produceErrorFigure(sys.argv[1],sys.argv[2])
filename = "twoTimeStepVarSigPOP.csv"
errorArray = np.genfromtxt(filename,delimiter = ',').transpose()
fileID = os.path.splitext(filename)[0]
errorArray1 = errorArray[0]
errorArray2 = errorArray[1]
errorArray3 = errorArray[2]
errorArray4 = errorArray[3]
errorArrayAVG = errorArray[11]

fig = plt.figure()
ax1 = fig.add_subplot(111)
xlabels = range(len(errorArray[0]))
#ax1.scatter(xlabels,errorArray1,s=5,c='b',marker='o')
#ax1.scatter(xlabels, errorArray2, s=5,c='r',marker='x')
#ax1.scatter(xlabels,errorArray3,s=5,c='r', marker='^')
#ax1.scatter(xlabels,errorArray4,s=5,c='g',marker = 'o')
ax1.scatter(xlabels,errorArrayAVG,s=5)
ax1.errorbar(xlabels,errorArrayAVG,yerr = errorArray[12],linestyle = 'None')
#plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.savefig("../../Main Writing/Figures/Errors/"+fileID+"Error.svg",format='svg')
plt.show()