from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class sortData():

	def __init__(self,minIdx = 1,maxIdx = 100):
		self.inputs = []
		self.outputs = []
		itrIdx = minIdx
		while itrIdx < maxIdx:
			temp = np.loadtxt("./Spike Results/1idTimes.csv", delimiter = ',' ).transpose()
			for i in range(len(temp)-1):
				self.inputs.append(temp[i])
				self.outputs.append(temp[i+1])
				itrIdx +=1
		self.batchID = 0
		popLen = len(temp)


	def next(self,batchSize):
		if self.batchId == len(self.data):
			self.batchId = 0
		batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
		batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
		self.batchId = min(self.batchId + batchSize, len(self.data))
		return batchData, batchLabels

testing = sortData(0,75)
#print(testing.inputs)
np.savetxt("testingInputs.csv",testing.inputs,delimiter = ",")
np.savetxt("testingOutputs.csv",testing.inputs,delimiter=",")