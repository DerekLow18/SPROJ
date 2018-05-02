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
def calculatePositives(groundTruth, hypothesizedMatrix,xc):
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
			if xc == True:
				if i != j:
					#count the true positives
					if groundTruth[i][j] ==1:
						if hypothesizedMatrix[i][j] == 1:
							tp += 1
						gtPositives += 1
					#count the false positives
					elif groundTruth[i][j] ==0:
						if hypothesizedMatrix[i][j] == 1:
							fp +=1
						gtNegatives +=1
			else:
				#count the true positives
				if groundTruth[i][j] ==1:
					if hypothesizedMatrix[i][j] == 1:
						tp += 1
					gtPositives += 1
				#count the false positives
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
def calculateROC(groundTruth, directory, matrixName,xc = False):
	tprList = []
	fprList = []
	#initialize the groundtruth as a np matrix
	groundTruth = np.genfromtxt(groundTruth, delimiter = ',')
	numFiles = 0
	#iterate through all files from xcThresholds, calculating the tpr and fpr for each
	for file in os.listdir(directory):
		if fnmatch.fnmatch(file,matrixName):
			#generate np matrix reflecting xc
			hypoMatrix = np.genfromtxt(directory+file,delimiter = ',')
			#np.fill_diagonal(hypoMatrix,0)
			tpr, fpr = calculatePositives(groundTruth,hypoMatrix,xc)
			tprList.append(tpr)
			fprList.append(fpr)
			numFiles += 1
	#return the sorted dictionary
	print(numFiles)
	return np.array(tprList), np.array(fprList)
'''
Resorting the array here to remove redundant tpr and fprs (a downsample with exactly the
same TPR and FPR rate.)
'''
def reSort(tpr,fpr):
	combined = np.column_stack((tpr,fpr))
	uniqueCombined = np.unique(combined,axis = 0)
	#print(uniqueCombined)
	uniques = np.hsplit(uniqueCombined,1)
	#print("after hsplit:",uniques)
	uniqueTPRs = np.squeeze(uniqueCombined[:,[0]])
	uniqueFPRs = np.squeeze(uniqueCombined[:,[1]])
	#print(uniqueTPRs,uniqueFPRs)
	return uniqueTPRs, uniqueFPRs

def locateIdealThreshold(tpr,fpr):
	currentMax = 0
	currentMaxID = 0
	for i in range(len(tpr)):
		if fpr[i] != 0:
			ratio = tpr[i]/fpr[i]
			if ratio > currentMax:
				currentMax = ratio
				currentMaxID = i
	return currentMax, currentMaxID, tpr[currentMaxID], fpr[currentMaxID]

if __name__=='__main__':
	#calculate the ROC plot
	defaultX = np.arange(0,1.01,0.01)
	defaultY = np.arange(0,1.01,0.01)
	groundTruth = "../Syn Weights/groundTruth10.csv"
	sumTprXC = np.array([])
	sumFprXC = np.array([])
	sumTprCOE= np.array([])
	sumFprCOE = np.array([])
	sumTprVar = np.array([])
	sumFprVar = np.array([])
	#cross correlation ROC
	aucXC = []
	for subdir in os.listdir("../Other Methods/xcThresholds/"):
		if fnmatch.fnmatch(subdir,"*xc"):
			#variable logistical activation function ROC
			tprXC, fprXC = calculateROC(groundTruth,"../Other Methods/xcThresholds/"+subdir+"/","*xcMatrix.csv",xc=True)
			if not sumTprXC.all():
				sumTprXC = tprXC
				sumFprXC = fprXC
			else:
				sumTprXC = np.append(sumTprXC,tprXC)
				sumFprXC = np.append(sumFprXC,fprXC)
		aucXC = np.append(aucXC,np.trapz(tprXC, fprXC))
	print("Finished with xc")
	aucVAR = []
	for subdir in os.listdir("../varSig thresholds/pop10/"):
		if fnmatch.fnmatch(subdir,"*downsample"):
			#variable logistical activation function ROC
			tprVar, fprVar = calculateROC(groundTruth,"../varSig thresholds/pop10/"+subdir+"/","*weightMatrix.csv")
			if not sumTprVar.all():
				sumTprVar = tprVar
				sumFprVar = fprVar
			else:
				sumTprVar = np.append(sumTprVar,tprVar)
				sumFprVar = np.append(sumFprVar,fprVar)
		aucVAR = np.append(aucVAR,np.trapz(tprVar, fprVar))
	print("Finished with varsig")
	aucCOE = []
	for subdir in os.listdir("../generalizedCOE thresholds/pop10/"):
		if fnmatch.fnmatch(subdir,"*downsample"):
			#generalized model ROC
			tprCOE, fprCOE = calculateROC(groundTruth,"../generalizedCOE thresholds/pop10/"+subdir+"/","*weightMatrix.csv")
			if not sumTprCOE.all():
				sumTprCOE = tprCOE
				sumFprCOE = fprCOE
			else:
				sumTprCOE = np.append(sumTprCOE,tprCOE)
				sumFprCOE = np.append(sumFprCOE,fprCOE)
		aucCOE = np.append(aucCOE,np.trapz(tprCOE, fprCOE))
	print("finished with COE")
	tprCOE, fprCOE = reSort(sumTprCOE,sumFprCOE)
	tprVar, fprVar = reSort(sumTprVar, sumFprVar)
	tprxc, fprxc = reSort(sumTprXC,sumFprXC)
	print(locateIdealThreshold(tprCOE,fprCOE))
	print(locateIdealThreshold(tprVar,fprVar))
	print(sumTprXC,sumFprXC)


	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	xcGraph = ax1.scatter(fprxc,tprxc,s=7,c='b',marker='o')
	defaultGraph = ax1.scatter(defaultX, defaultY, s=7,c='m',marker='x')
	COEGraph = ax1.scatter(fprCOE,tprCOE,s=7,c='r', marker='^')
	varGraph = ax1.scatter(fprVar,tprVar,s=7,c='g',marker = '|')
	ax1.legend((defaultGraph,xcGraph,COEGraph,varGraph),("Linear","XC","Base","Activation"))
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	plt.savefig("../../Main Writing/Figures/rocPop10.svg",format='svg')
	plt.show()
	#print(tprxc,fprxc)
	print("xc:",np.mean(aucXC),np.std(aucXC),np.trapz(tprxc,fprxc))
	print("norm:",np.trapz(defaultY,defaultX))
	print("var:",np.mean(aucVAR),np.std(aucVAR),np.trapz(tprVar, fprVar))
	print("COE:",np.mean(aucCOE),np.std(aucCOE),np.trapz(tprCOE,fprCOE))
	#plot them using pyplot
