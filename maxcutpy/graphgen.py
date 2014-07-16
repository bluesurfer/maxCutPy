"""
This library contains methods for generating benchmarks which will be used
for testing.

"""

import os
import math
import cPickle as pkl
from collections import defaultdict

import networkx as nx


__license__ = "GPL"


DEFAULT_FILE_LOCATION = os.path.abspath(os.curdir) + "/testing/"
BENCHMARKS_FILE_DIR = 'benchmarks'


#==============================================================================
# File Management
#==============================================================================


def write_to_file(obj, filename, location=''):
    """Write object to file using pickle."""
    directory = DEFAULT_FILE_LOCATION + location + '/'

    if not os.path.exists(directory):
        print('Creating directory: ' + str(directory))
        os.makedirs(directory)

    pkl_file = open(directory + filename, 'w')
    pkl.dump(obj, pkl_file)
    pkl_file.close()


def read_from_file(filename, location=''):
    """Read object from file using pickle."""

    file_path = DEFAULT_FILE_LOCATION + location + '/' + filename

    # check if the file exists
    if not os.path.isfile(file_path):
        print('ERROR: \"' + filename + '\"in \"' + location + '\" is missing.')
        exit()

    pkl_file = open(file_path, 'r')
    obj = pkl.load(pkl_file)
    pkl_file.close()
    return obj


def merge_dictionaries(dicts_of_list):
    """Merge a bunch of dictionaries of list."""
    merged_dict = defaultdict(list)

    for d in dicts_of_list:
        for key in d.keys():
            merged_dict[key].extend(d[key])

    return merged_dict


def dictionary_size(dict_of_list):

    size = 0

    for key in dict_of_list:
        size += len(dict_of_list[key])

    return size


def remove_unconnected_graphs(eps_dict):
    filtered_eps_dict = defaultdict(list)
    n = dictionary_size(eps_dict)

    i = 0

    for eps in eps_dict:
        for A, c in eps_dict[eps]:
            G = nx.from_numpy_matrix(A)
            if nx.is_connected(G):
                print(i)
                filtered_eps_dict[eps].append((A, c))
            else:
                print(str(i) + ' removing..')
            i += 1

    m = dictionary_size(filtered_eps_dict)
    print('Graphs removed: ' + str(n - m))
    return filtered_eps_dict


#==============================================================================
# Benchmarks Building Methods
#
# WARNING: if a file already exists it will be automatically overwrited.
#
# Note
# -----
# In this project we treat only connected graphs. So, in these benchmark's
# generation's methods if a graph is unconnected it will be ignored.
#==============================================================================


def generate_static_graphs(n_graphs, n_nodes, p_edges):
    """Creates a set of n graphs with n nodes and choice of possible edges
    with probability p.

    """
    filename = 'static_' + str(n_graphs) + 'G_' + str(n_nodes) + 'N_'
    filename += str(p_edges) + 'P'

    adj_matrix_dict = {}

    for i in range(0, n_graphs):

        G = nx.erdos_renyi_graph(n_nodes, p_edges)

        if nx.is_connected(G):
            adj_matrix_dict[i] = nx.adj_matrix(G)

    write_to_file(adj_matrix_dict, filename + '.dat', BENCHMARKS_FILE_DIR)

    return filename


def generate_crescent_edges_graphs(n_graphs, n_nodes, p_edges, min_p_edges=None):
    """Creates a set of n graphs with a fixed n nodes and choice of
    possible edges with crescent probability p.

    """

    adj_matrix_dict = defaultdict()

    if min_p_edges:
        min_p_edges = math.log1p(n_nodes) / n_nodes

    filename = 'edges_' + str(n_graphs) + 'G_' + str(n_nodes) + 'N'
    filename += '_from_' + str(min_p_edges) + '_to_' + str(p_edges) + 'P'

    j = (p_edges - min_p_edges) / n_graphs  # p increment

    p_edges = min_p_edges

    for i in range(0, n_graphs):
    
        G = nx.erdos_renyi_graph(n_nodes, p_edges)
    
        if nx.is_connected(G):
            adj_matrix_dict[p_edges] = nx.adj_matrix(G)

        p_edges += j

    write_to_file(adj_matrix_dict, filename + '.dat', BENCHMARKS_FILE_DIR)

    return filename


def generate_crescent_nodes_graphs(nodes_range, p_edges):
    """Creates a set of graphs in a nodes range with choice of possible edges
    with probability p.

    """
    file_info = 'from_' + str(min(nodes_range)) + '_to_' + str(max(nodes_range))
    file_info += 'N_' + str(p_edges) + 'P'

    adj_matrix_dict = defaultdict(list)

    for nodes in nodes_range:

        G = nx.erdos_renyi_graph(nodes, p_edges)

        if nx.is_connected(G):
            adj_matrix_dict[nodes] = nx.adj_matrix(G)

    filename = 'nodes_' + file_info
    write_to_file(adj_matrix_dict, filename + '.dat', BENCHMARKS_FILE_DIR)

    return filename

