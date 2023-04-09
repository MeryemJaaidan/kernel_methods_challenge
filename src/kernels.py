import numpy as np
import networkx as nx
from scipy import sparse

def K_n_walk(data_set, n=2):
    d = len(data_set)
    K = np.zeros((d,d))
    for i in range(d):
        for j in range(i+1):
            G = nx.tensor_product(data_set[i], data_set[j])
            A = nx.adjacency_matrix(G)
            K[i,j] = (A ** n).sum()
            K[j,i] = K[i,j]
    
    return K

# def K_random_walk(G1, G2):
#     '''random walk kernel'''

#     return K


# def K_geometric_walk(G1, G2):
#     '''geometric walk kernel'''
#     return K
