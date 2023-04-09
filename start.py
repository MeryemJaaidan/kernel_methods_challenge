"""
training_data.pkl contains a list of size 5000. Each element of the list is an object of the class networkx.classes.graph.Graph from the package networkx. For instance, if G is a graph in the list, then G represents a molecule.

Nodes of the graph G represent atoms forming the corresponding molecule (ex. node = G.nodes[0] is the first atom of graph G). The atom type is encoded in the field 'labels' (node['labels']) for each node as a list of size 1 containing an integer with values ranging from 0 to 49.

Edges of the graph G represent bonds between two atoms of the molecules (ex. edge= G.edges[0,1] is the bond (when it exists) between atom 0 and atom 1 of the molecule). 
The bond type is encoded in the field 'labels' (edge['label']) for each edge as a list of size 1 containing an integer with values ranging from 0 to 3.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from src.utils import *
from src.models import kernelSVM
from src.kernels import K_n_walk


parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'data/training_data.pkl')
parser.add_argument('--method', type = str, default = 'KernelSVM')

args = parser.parse_args()

print('-----Options-----')
print(args)
print('-----End-----')

# Load data
training = pd.read_pickle(args.data)
labels = pd.read_pickle('data/training_labels.pkl')

# training only on a part of training data to reduce the Kernel computation time
indx = np.arange(100)
train_data = training[indx] 
train_labels = labels[indx]
val_data = pd.read_pickle('data/test_data.pkl')[indx] 

## Compute Kernel
print("----- Computing Kernel ------")
begin = time.time()
K_train = K_n_walk(train_data)
K_test = K_n_walk(val_data)

end = time.time()
print("----- Kernels computed in {:.2f} minutes ------".format((end - begin)/60))

## Train SVM model
clf = kernelSVM(lmbd=0.01, loss='squared_hinge',reformated = True)
clf.train(K_train, train_labels)

## Predict output
y_pred = clf.predict(K_test)

print("----- Labels predicted ------\n", y_pred)