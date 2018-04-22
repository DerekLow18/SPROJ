import sys, os, fnmatch, re
import numpy as np
import matplotlib.pyplot as plt
import re

#produce a scatter plot from a csv file, depicting error changes
if __name__=='__main__':
	if len(sys.argv) < 2:
		raise Exception("Please provide a csv file of the error.")
		exit()
	errorArray = np.genfromtxt(str(sys.argv[1]),delimiter = ',')
	fileID = os.path.splitext(str(sys.argv[1]))[0]
	plt.scatter(range(len(errorArray)),errorArray,s=5)
	plt.xlabel("Iteration")
	plt.ylabel("Mean Squared Error")
	plt.savefig("../../Main Writing/Figures/"+fileID + "Error.svg",format = 'svg')
	plt.show()