'''
ROC curves for the different methods of reconstruction.
'''
import sys, os, fnmatch, re
import numpy as np
import matplotlib.pyplot as plt
'''
calculate true and false positive values

we define the TPR as # true positives/ # of ground truth positives
we define the FPR as # false positives/ # of ground truth negatives

arguments: two numpy matrices of n x n dimensionality

'''
def calculatePositives(groundTruth, hypothesizedMatrix):
	#first, we keep a count of the number of true positives and false positives
	tp = 0
	fp = 0
	gtPositives=0
	gtNegatives=0
	#we need to iterate through the hypothesized matrix and compare to the ground truth matrix
	#if both are equal and 1, increase the tp count. if hypothezied is 1 and groundTruth is 0,
	#add 1 to fp
	for i in range(len(hypothesizedMatrix)):
		for j in range(len(hypothesizedMatrix)):
			if groundTruth[i][j] ==1:
				if hypothesizedMatrix[i][j] == 1:
					tp += 1
				gtPositives += 1
			elif groundTruth[i][j] ==0:
				if hypothesizedMatrix[i][j] == 1:
					fp +=1
				gtNegatives +=1
	#now we divide tp by the number of positives in ground truth,
	#and we divide fp by number of negatives in ground truth
	tpr = tp/gtPositives
	fpr = fp/gtNegatives
	return tpr, fpr
'''
calculate all tpr and fpr for the given two matrices
1. in a for loop, calculate and store all tpr and fpr values
2. order and remove redundant rates
3. plot each point
'''
def calculateROCPlot():
	#gonna hard code the files first to ensure that it works
	#create a dictionary that will store the tpr and fpr

	#initialize the groundtruth as a np matrix
	groundTruth = np.genfromtxt("../Syn Weights/groundTruth1.csv", delimiter = ',')
	#iterate through all files from xcThresholds, calculating the tpr and fpr for each
	for file in os.listdir("../Other Methods/xcThresholds"):
		if fnmatch.fnmatch(file,"*xcMatrix.csv"):
			#generate np matrix reflecting xc
			hypoMatrix = np.genfromtxt(file,delimiter = ',')
			tpr, fpr = calculatePositives(groundTruth,hypoMatrix)

	#sort the dictionary from lowest tpr/fpr to highest tpr/fpr

	#return the sorted dictionary
	return

if __name__=='__main__':
	#calculate the ROC plot

	#plot them using pyplot
