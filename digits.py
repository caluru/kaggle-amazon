import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

filepath = '/mnt/c/Users/chait/Downloads/digits/'

def printAsciiDigit(digit):
	for i in range(28):
		for j in range(28):
			print '0' if digit[28*i+j] > 0 else ' ',
		print ''

def printDigit(digit):
	image = np.reshape(digit, (28, 28))
	plt.imshow(image)
	plt.show()

#Extract Train Data
f = open(filepath + 'train.csv','r')
r = csv.reader(f, delimiter = ',')
train = []
trainLabels = []
first = True
for row in r:
	if first:
		first = not first
		continue
	train.append(np.array([int(i) for i in row[1:]]))
	trainLabels.append(int(row[0]))
trainLabels = np.array(trainLabels)

#Extract Test Data
test = []
f = open(filepath + 'test.csv','r')
r = csv.reader(f, delimiter = ',')
test = []
first = True
for row in r:
	if first:
		first = not first
		continue
	test.append(np.array([int(i) for i in row]))


