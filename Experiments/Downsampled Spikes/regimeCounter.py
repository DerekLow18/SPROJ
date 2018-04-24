import numpy as np
import os, sys
import fnmatch
import csv

regimeDict = {}
path = "./pop10/"
for file in os.listdir(path):
	if fnmatch.fnmatch(file,"*downsample.csv"):
		spikeTime = np.genfromtxt(path+file,delimiter = ',')
		for i in range(len(spikeTime)):
			sumSpikes = np.sum(spikeTime[i])
			if sumSpikes in regimeDict:
				regimeDict[sumSpikes] += 1
			else:
				regimeDict[sumSpikes] = 1

w = csv.writer(open("regimeDict.csv", "w"))
for key, val in regimeDict.items():
	w.writerow([key, val])