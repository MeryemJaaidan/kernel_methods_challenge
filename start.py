import pandas as pd
import numpy as np
#from src.models import kernelSVM
from src.methods.kernelSupportVectorMachine import KernelSVM
from src.kernels import kernel_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'data/training_data.pkl')
parser.add_argument('--method', type = str, default = 'KernelSVM')

args = parser.parse_args()

print('-----Options-----')
print(args)
print('-----End-----')

training = pd.read_pickle('data/training_data.pkl')
labels = pd.read_pickle('data/training_labels.pkl')
val_data = pd.read_pickle('data/test_data.pkl')

K_train = kernel_matrix(training)
K_test = kernel_matrix(val_data)

# Uses the SVM classifier to perform classification
clf = KernelSVM(C=1, kernel='rbf')
clf.train(K_train, labels)
y_pred = clf.predict(K_test)

#save the results in test_pred.csv
np.savetxt('test_pred.csv', y_pred, delimiter=',')