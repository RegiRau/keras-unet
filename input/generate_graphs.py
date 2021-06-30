"""
======================
Generate Random Geometric Graph for Training
======================
"""
import networkx as nx
from networkx.algorithms import bipartite

from scipy import sparse
import matplotlib.pyplot as plt

import numpy as np

import sys
from skimage import color


import os
import pkgutil
search_path = ['.'] # set to None to see all modules importable from sys.path
all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
print(all_modules)

sys.path.append('../')


def get_position_vector(pos_list):
    pos_list
    pos = np.empty((len(pos_list),2))
    for key in pos_list.keys():
        pos[key]= np.array(pos_list[key])
    return pos


def get_random_attributes(G):
    np.random.seed(2021)
    G.node_attributes = np.random.rand(G.N)


def generate_graph_data(Graph, location_name = 'test_graph'):
    # position is stored as node attribute data for random_geometric_graph
    pos = nx.get_node_attributes(Graph, "pos")
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(Graph, pos, alpha=0.4)
    nx.draw_networkx_nodes(
        Graph,
        pos,
        node_size=80,
        )
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.savefig(location_name)
    plt.show()


    pos_list = pos
    adjacency = nx.adjacency_matrix(Graph)
    position = get_position_vector(pos_list)
    output = np.concatenate((position, adjacency.todense()), axis=1)
    return output


adjcoutput_matrix = {}
for i in range(0,100):
    #Graph = nx.random_geometric_graph(35, 0.3, seed=896803)
    Graph = nx.random_geometric_graph(35, 0.21)
    data_name = 'graph_images/train/image/' + str(i).zfill(5) + 'graph'
    output = generate_graph_data(Graph, location_name = data_name)
    print('training output nr', i, ':', output )

    adjcoutput_matrix[str(i).zfill(5)] = output

np.save('graph_images/train/label/adjcouput_matrix', adjcoutput_matrix )

for i in range(0,10):
    #Graph = nx.random_geometric_graph(35, 0.3, seed=896803)
    Graph = nx.random_geometric_graph(35, 0.21)
    data_name = 'graph_images/test/image/' + str(i).zfill(5) + 'graph'
    output = generate_graph_data(Graph, location_name = data_name)
    print('training output nr', i, ':', output )

    adjcoutput_matrix[str(i).zfill(5)] = output

np.save('graph_images/test/label/adjcouput_matrix', adjcoutput_matrix )

