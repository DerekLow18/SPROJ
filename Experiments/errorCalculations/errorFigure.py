import sys, os, fnmatch, re
import numpy as np
import matplotlib.pyplot as plt

#produce a scatter plot from a csv file, depicting error changes
if __name__=='__main__':
	if len(sys.argv) < 2:
		raise Exception("Please provide a csv file of the error.")
		exit()
	errorArray = np.genfromtxt(str(sys.argv[1]),delimiter = ',')
	plt.scatter(range(len(errorArray)),errorArray,s=5)
	plt.xlabel("Iteration")
	plt.ylabel("Mean Squared Error")
	plt.show()