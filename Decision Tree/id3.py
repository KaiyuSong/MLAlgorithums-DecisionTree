#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
from __future__ import division
import sys
import re
import math
# Node class for the decision tree
import node


train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
	if p == 1 or p == 0:
		result = 0;
	else:
		rest = 1 - p;
		result = - (p * math.log(p, 2)) - (rest * math.log(rest, 2));
	return result
	
# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
	main = entropy(py/total);
	if pxi:
		plussituation = entropy(py_pxi/pxi)
	else:
		plussituation = 0;
	if pxi != total:
		minussituation = entropy((py - py_pxi)/(total - pxi));
	else:
		minussituation = 0;
	result = main - (pxi/total)*plussituation - ((total-pxi)/total)*minussituation;
	return result

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable	
	
	
	
# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
		data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
	total = len(data);

	py = getpy(data);
	if py == 0:
		return node.Leaf(varnames, 0)
	elif py == total:
		return node.Leaf(varnames, 1)
	else:
		(positivedata, negativedata) = splitdata(data, varnames);
		index = findbestindex(data, varnames);
		if index == -1:
			return node.Leaf(varnames, 0)
		return node.Split(varnames, index, build_tree(negativedata, varnames), build_tree(positivedata, varnames))
	return None

def getpy(data):
	py = 0;
	for i in xrange(0,len(data)):
		if (data[i][-1]==1):
			py += 1;
	return py;

def getpxi(data, varnames):
	pxi = [0 for i in range(len(varnames)-1)]
	for x in xrange(0, len(data)):
		for i in xrange(0,(len(varnames)-1)):
			if data[x][i]==1:
				pxi[i] += 1;
	return pxi;

def getpypxi(data, varnames):
	pypxi = [0 for i in range(len(varnames)-1)]
	for x in xrange(0, len(data)):
		for i in xrange(0,(len(varnames)-1)):
			if data[x][i]==data[x][-1]==1:
				pypxi[i] += 1;
	return pypxi;

def findbestindex(data, varnames):
	py = getpy(data);
	pxi = getpxi(data, varnames);
	pypxi = getpypxi(data, varnames);
	kind = len(varnames) - 1;
	index = 0;
	for i in xrange(0,kind):
		if pxi[i] != 0:
			new = infogain(pypxi[i] ,pxi [i], py, len(data));
			old = infogain(pypxi[index] ,pxi [index], py, len(data));
			if (new > old):
				index = i;
	if infogain(pypxi[index], pxi[index], py, len(data)) == 0:
		return -1
	return index;

def splitdata(data, varnames):
	positivedata = [];
	negativedata = [];
	bestindex = findbestindex(data, varnames);
	if bestindex == -1:
		return ([],data) 
	for i in xrange(0,len(data)):
		if data[i][bestindex] == 1:
			positivedata.append(data[i]);
		else:
			negativedata.append(data[i]);
		
	return (positivedata, negativedata);

# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
	global train
	global varnames
	global test
	global testvarnames
	global root
	(train, varnames) = read_data(trainS)
	(test, testvarnames) = read_data(testS)
	modelfile = modelS
	
	# build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
	root = build_tree(train, varnames)
	print_model(root, modelfile)
	
def runTest():
	correct = 0
	# The position of the class label is the last element in the list.
	yi = len(test[0]) - 1
	for x in test:
		# Classification is done recursively by the node class.
        # This should work as-is.
		pred = root.classify(x)
		if pred == x[yi]:
			correct += 1
	acc = float(correct)/len(test)
	return acc	
	
	
# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
		print 'Usage: id3.py <train> <test> <model>'
		sys.exit(2)
    loadAndTrain(argv[0],argv[1],argv[2]) 
                    
    acc = runTest()             
    print "Accuracy: ",acc                      

if __name__ == "__main__":
    main(sys.argv[1:])
