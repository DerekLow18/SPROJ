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
def calculateROC(groundTruth, directory, matrixName):
	tprList = []
	fprList = []
	#initialize the groundtruth as a np matrix
	groundTruth = np.genfromtxt(groundTruth, delimiter = ',')
	#iterate through all files from xcThresholds, calculating the tpr and fpr for each
	for file in os.listdir(directory):
		if fnmatch.fnmatch(file,matrixName):
			#generate np matrix reflecting xc
			hypoMatrix = np.genfromtxt(directory+file,delimiter = ',')
			np.fill_diagonal(hypoMatrix,0)
			tpr, fpr = calculatePositives(groundTruth,hypoMatrix)
			tprList.append(tpr)
			fprList.append(fpr)
	#return the sorted dictionary
	return np.array(tprList), np.array(fprList)

def reSort(tpr,fpr):
	combined = np.column_stack((tpr,fpr))
	uniqueCombined = np.unique(combined,axis = 0)
	print(uniqueCombined)
	uniques = np.hsplit(uniqueCombined,1)
	print("after hsplit:",uniques)
	uniqueTPRs = np.squeeze(uniqueCombined[:,[0]])
	uniqueFPRs = np.squeeze(uniqueCombined[:,[1]])
	print(uniqueTPRs,uniqueFPRs)
	return uniqueTPRs, uniqueFPRs

if __name__=='__main__':
	#calculate the ROC plot
	defaultX = np.arange(0,1,0.01)
	defaultY = np.arange(0,1,0.01)
	tprxc,fprxc = calculateROC("../Syn Weights/groundTruth10.csv","../Other Methods/xcThresholds/","*xcMatrix.csv")
	tprxc,fprxc = reSort(tprxc,fprxc)
	#tprxc,fprxc = calculateROCxc()
	tprCOE, fprCOE = calculateROC("../Syn Weights/groundTruth10.csv","../generalizedCOE thresholds/","*weightMatrix.csv")
	tprCOE, fprCOE = reSort(tprCOE,fprCOE)
	#tprCOE,fprCOE = calculateROCgCOE()
	tprVar, fprVar = calculateROC("../Syn Weights/groundTruth10.csv","../varSig thresholds/","*weightMatrix.csv")
	tprVar, fprVar = reSort(tprVar, fprVar)
	#tprVar, fprVar = calculateROCvarSig()
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.scatter(fprxc,tprxc,s=5,c='b',marker='o')
	ax1.scatter(defaultX, defaultY, s=5,c='r',marker='x')
	ax1.scatter(fprCOE,tprCOE,s=5,c='r', marker='^')
	ax1.scatter(fprVar,tprVar,s=5,c='g',marker = 'o')
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()
	print("xc:",np.trapz(fprxc,tprxc))
	print("norm:",np.trapz(defaultY,defaultX))
	print("var:",np.trapz(fprVar,tprVar))
	#plot them using pyplot
