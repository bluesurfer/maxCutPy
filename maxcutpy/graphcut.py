"""
Functional library with graph cut methods and assorted utilities.

"""

import random

import networkx as nx
import numpy as np


__license__ = "GPL"


# Graph labels
PARTITION = 'partition'

# Node's classes
BLUE = 1
BLACK = -1

UNDECIDED = 0  # magenta
MARKED = 2     # red


#==============================================================================
# Graph's partitions
#==============================================================================


def partition_dictionary(G):
    """Return a dictionary rapresentation of a cut"""
    return nx.get_node_attributes(G, PARTITION)


def set_partitions(G, blue_nodes, black_nodes):
    """Set node's blue class and black class"""

    init_cut(G)
    cut(G, dict.fromkeys(blue_nodes, BLUE))
    cut(G, dict.fromkeys(black_nodes, BLACK))


def get_partitions(G, nbunch=None):
    """Return all partitions of a graph G as different sets"""

    if nbunch is None:
        nbunch = G.nodes()

    blue_nodes = set()
    black_nodes = set()
    undecided_nodes = set()
    marked_nodes = set()

    for i in nbunch:

        if G.node[i][PARTITION] is BLUE:
            blue_nodes.add(i)

        elif G.node[i][PARTITION] is BLACK:
            black_nodes.add(i)

        elif G.node[i][PARTITION] is MARKED:
            marked_nodes.add(i)

        else:
            undecided_nodes.add(i)

    return (blue_nodes, black_nodes, undecided_nodes, marked_nodes)


def all_possible_cuts(G):
    """Return all possible cut graphs.

    Warning: demonstration porpuse only

    """
    cuts_list = []
    n = G.number_of_nodes()

    for i in range(1, 2 ** (n - 1)):
        cut_graph = nx.Graph(G)
        binary_cut(cut_graph, i)
        cuts_list.append(cut_graph)

    return cuts_list


#==============================================================================
# Cut Indices
#==============================================================================


def are_undecided_nodes(G):
    """Check the existence of undecided nodes"""
    for v in G.nodes():
        if G.node[v][PARTITION] == UNDECIDED:
            return True
    return False


def edges_beetween(G, a, b):
    """Return the number of edges between two sets of nodes

    WARNING: a and b should have no element in common.

    """
    return len(nx.edge_boundary(G, a, b))


def cut_edges(G, partition_dict=None):
    """Return the value of the cut.
    Cut edges: the number of cross edges between blue and black nodes.

    """
    if partition_dict is not None:
        nbunch = partition_dict.keys()
        cut(G, partition_dict)
    else:
        nbunch = G.nodes()

    blue_nodes, black_nodes, undecided = get_partitions(G, nbunch)[0:3]
    return edges_beetween(G, blue_nodes, black_nodes)


def compute_epsilon(G):
    """Compute epsilon value of a cut graph.

    Epsilon := 1 - X / |E| where X is the number of cut edges

    """
    return round(1.0 - float(cut_edges(G)) / float(G.number_of_edges()), 3)


#==============================================================================
# Consistency Condition
#==============================================================================


def minority_class(G, partition_dict, node):
    """Compute the minority class of a node.

    Minority class: tells the class of which a node should belong
    in order to maximize the cut.

    """
    neighbors = G.neighbors(node)

    blue_neighbors = 0
    black_neighbors = 0

    size = len(neighbors)

    for i in neighbors:
        if partition_dict[i] is BLUE:
            blue_neighbors += 1
        elif partition_dict[i] is BLACK:
            black_neighbors += 1

    if blue_neighbors > size / 2:
        return BLACK

    if black_neighbors > size / 2:
        return BLUE

    return UNDECIDED


def strong_minority_class(G, partition_dict, node):
    """Compute strong minority class of a node"""
    neighbors = G.neighbors(node)

    blue_neighbors = 0
    black_neighbors = 0

    for i in neighbors:
        if partition_dict[i] is BLUE:
            blue_neighbors += 1
        elif partition_dict[i] is BLACK:
            black_neighbors += 1

    # if all neighbors are marked
    if blue_neighbors == 0 and black_neighbors == 0:
        return MARKED

    if blue_neighbors > black_neighbors:
        return BLACK
    return BLUE


def is_cut_consistent(G, partition_dict, check_nodes=None):
    """Check cut consistency condition."""
    if check_nodes is None:
        check_nodes = partition_dict.keys()

    for i in check_nodes:
        node_class = minority_class(G, partition_dict, i)
        if partition_dict[i] is not MARKED:
            if (node_class is not UNDECIDED and
                partition_dict[i] is not UNDECIDED):
                if node_class != partition_dict[i]:
                    return False

    return True


#==============================================================================
# Cut Methods
#==============================================================================


def init_cut(G, nbunch=None):
    """Initialize cut: set all nodes or a bunch of nodes as undecided"""
    if nbunch is None:
        nbunch = G.nodes()

    nx.set_node_attributes(G, PARTITION, dict.fromkeys(nbunch, UNDECIDED))


def integer_to_binary(i, n):
    """Convert an integer to binary."""
    rep = bin(i)[2:]
    return ('0' * (n - len(rep))) + rep


def cut(G, partition_dict):
    """Use a partition dictionary to cut a graph"""
    nx.set_node_attributes(G, PARTITION, partition_dict)


def binary_cut(G, int_cut, bin_cut=None):
    """Cut a graph G using a binary operation."""
    if bin_cut is None:
        bin_cut = integer_to_binary(int_cut, G.number_of_nodes())

    for i, node in enumerate(G.nodes()):
        if bin_cut[i] is '0':
            G.node[node][PARTITION] = BLACK
        else:
            G.node[node][PARTITION] = BLUE

    return nx.get_node_attributes(G, PARTITION)


def could_be_cut(G, partition_dict):
    """Return true if the graph is being cut according
    to the node's minority class.

    """
    is_cut = False

    for i in partition_dict:
        if partition_dict[i] is UNDECIDED or partition_dict[i] is MARKED:
            node_color = minority_class(G, partition_dict, i)
            if node_color is not UNDECIDED:
                partition_dict[i] = node_color
                is_cut = True

    return is_cut


def marked_nodes_could_be_cut(G, partition_dict, marked_nodes):
    """Try to cut the marked nodes using the strong minority class
    for each node

    """
    buffer_dict = dict(partition_dict)

    for i in marked_nodes:
        if partition_dict[i] is MARKED:
            node_class = strong_minority_class(G, buffer_dict, i)
            partition_dict[i] = node_class


#==============================================================================
# Marking Strategies Methods
#==============================================================================


def degree_nodes_sequence(G, nbunch=None, reverse=False):
    """Return a list of nodes in ascending order according to
    their degree value.

    """
    degrees_dict = G.degree(nbunch)
    return sorted(degrees_dict,
                  key=lambda key: degrees_dict[key],
                  reverse=reverse)


def lowest_degree_nodes(G, n_nodes):
    """Return the n nodes with lowest degree"""
    deg_node_seq = degree_nodes_sequence(G, reverse=False)
    return deg_node_seq[0:n_nodes]


def highest_degree_nodes(G, n_nodes):
    """Return the n nodes with highest degree"""
    deg_node_seq = degree_nodes_sequence(G, reverse=True)
    return deg_node_seq[0:n_nodes]


def two_maximal_independent_set(G):
    """Return a set of nodes from a bipartite subgraph"""
    i0 = nx.maximal_independent_set(G)
    i1 = nx.maximal_independent_set(G, i0)

    b = set(i0) | set(i1)

    return b


def pick_random_nodes(G, n_nodes):
    """Return a random set of n nodes from the graph"""
    return random.sample(G.nodes(), n_nodes)


#==============================================================================
# Others
#==============================================================================


def is_all_isolate(G):
    """Return true if all the nodes in the graph G are isolates"""

    nodes = G.nodes()

    if not nodes:
        return False

    for v in nodes:
        if nx.degree(G, v) != 0:
            return False
    return True


def remove_isolates_nodes(G):
    """Remove isolated nodes from a graph G.

    Isolated node: node with degree equal to zero.

    """
    isolate = False
    if not nx.is_connected(G):
        for node in G.nodes():
            if G.degree(node) == 0:
                G.remove_node(node)
                isolate = True
    return isolate


def sign_norm(d):
    """Normalize dictionary keys according to sign function"""
    sign_d = {}
    for i in d:
        sign_d[i] = np.sign(d[i])
    return sign_d

