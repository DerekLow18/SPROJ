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
def calculateROCxc():
	#gonna hard code the files first to ensure that it works
	#create a dictionary that will store the tpr and fpr
	tprList = []
	fprList = []
	#initialize the groundtruth as a np matrix
	groundTruth = np.genfromtxt("../Syn Weights/groundTruth10.csv", delimiter = ',')
	#iterate through all files from xcThresholds, calculating the tpr and fpr for each
	for file in os.listdir("../Other Methods/xcThresholds"):
		if fnmatch.fnmatch(file,"*xcMatrix.csv"):
			#generate np matrix reflecting xc
			hypoMatrix = np.genfromtxt("../Other Methods/xcThresholds/"+file,delimiter = ',')
			tpr, fpr = calculatePositives(groundTruth,hypoMatrix)
			tprList.append(tpr)
			fprList.append(fpr)
			print(tpr,fpr)
	#return the sorted dictionary
	return np.array(tprList), np.array(fprList)

def calculateROCgCOE():
	print("calculating COE")
	tprList = []
	fprList = []
	#initialize the groundtruth as a np matrix
	groundTruth = np.genfromtxt("../Syn Weights/groundTruth10.csv", delimiter = ',')
	#iterate through all files from xcThresholds, calculating the tpr and fpr for each
	for file in os.listdir("../generalizedCOE thresholds"):
		if fnmatch.fnmatch(file,"*weightMatrix.csv"):
			#generate np matrix reflecting xc
			hypoMatrix = np.genfromtxt("../generalizedCOE thresholds/"+file,delimiter = ',')
			np.fill_diagonal(hypoMatrix,0)
			tpr, fpr = calculatePositives(groundTruth,hypoMatrix)
			tprList.append(tpr)
			fprList.append(fpr)
	#return the sorted dictionary
	return np.array(tprList), np.array(fprList)

if __name__=='__main__':
	#calculate the ROC plot
	defaultX = np.arange(0,1,0.01)
	defaultY = np.arange(0,1,0.01)
	tprxc,fprxc = calculateROCxc()
	tprCOE,fprCOE = calculateROCgCOE()
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.scatter(fprxc,tprxc,s=5,c='b',marker='o')
	ax1.scatter(defaultX, defaultY, s=5,c='r',marker='x')
	ax1.scatter(fprCOE,tprCOE,s=5,c='r', marker='^')
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()
	print("xc:",np.trapz(fprxc,tprxc))
	print("norm:",np.trapz(defaultY,defaultX))
	#plot them using pyplot
