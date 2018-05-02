import numpy as np
import sys

if len(sys.argv) < 2:
	print("Incorrect number of arguments. Please enter the file name.")
	exit()

filename = str(sys.argv[1])
dataset = np.genfromtxt(filename,delimiter = ',')
dataset = dataset.transpose()
print(dataset.shape)
print(len(dataset))
newDataset = []
for i in range(len(dataset)):
	if dataset[i].any():
		print(dataset.shape)
		newDataset.append(dataset[i])

print(len(newDataset))
np.savetxt("downsampleNoEmpty.csv",newDataset,delimiter=',')