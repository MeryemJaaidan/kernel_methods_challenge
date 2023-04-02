import numpy as np
import networkx as nx
from scipy import sparse

def K_n_walk(G1, G2, n=2):
    G = nx.tensor_product(G1, G2)
    A = nx.adjacency_matrix(G)
    K = (A ** n).sum()
    return K

# def K_random_walk(G1, G2):
#     '''random walk kernel'''

#     return K


# def K_geometric_walk(G1, G2):
#     '''geometric walk kernel'''
#     return K
