import numpy as np

# Data preprocessing

def convert_to_numpy(df):
    return df.to_numpy()

def drop_coloumns(df, ids):
    return df.drop(ids, axis=1)

def squeeze(df):
    return df.squeeze()

# Model evaluation

def get_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

def precision(y_pred, y_true):
    if np.sum(y_pred) == 0:
        return 0
    else:
        return np.sum(y_pred * y_true) / np.sum(y_pred)
    
def recall(y_pred, y_true):
    if np.sum(y_true) == 0:
        return 0
    else:
        return np.sum(y_pred * y_true) / np.sum(y_true)
    
def f1_score(y_pred, y_true):
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    if p + r == 0:
        return 0
    else:
        return 2 * p * r / (p + r)
    
# Kernels
# List of kernels: https://scikit-learn.org/stable/modules/metrics.html#metrics

def linear_kernel(X, Y):
    """
    Return the linear kernel between X and Y::
    """
    return np.dot(X, Y.T)