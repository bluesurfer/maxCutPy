"""
Functional library with maximum cut methods to a graph and other auxiliary
function.

CAUTION: some methods may require a heavy computational load.

"""

import signal
from multiprocessing import Process, Queue, cpu_count

import networkx as nx
import numpy as np
from scipy import integrate

import graphcut as gc


__license__ = 'GPL'


#==============================================================================
# Time out decorator
#==============================================================================


class TimedOutExc(Exception):
    pass


def timeout(timeout):
    def decorate(f):
        def handler(signum, frame):
            raise TimedOutExc()

        def new_f(*args, **kwargs):

            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)

            try:
                result = f(*args, **kwargs)
            except TimedOutExc:
                result = None
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
            return result

        new_f.func_name = f.func_name
        return new_f

    return decorate


#==============================================================================
# Theoretic Approximation Functions
#
# Implementation of Soto's and Trevisan's approximation functions.
#==============================================================================


def f1(x, eps):
    return ((-1.0 + np.sqrt(4.0 * (eps / x) ** 2 - 8.0 * eps / x + 5.0)) /
            (2.0 * (1.0 - eps / x)))


def f2(x, eps):
    return (1.0 / (1.0 + 2.0 * np.sqrt((1.0 - eps / x) * eps / x)))


def soto_function(eps):

    eps0 = 0.2280155  # unique solution

    if eps >= 1.0 / 3.0:
        return 1.0 / 2.0

    if eps >= eps0 and eps <= 1.0 / 3.0:
        return ((integrate.quad(lambda x: 1.0 / 2.0, 0, 3 * eps)[0] +
                 integrate.quad(f1, 3.0 * eps, 1.0, eps)[0]
                 ))

    if eps == 0:
        return 1.0

    if eps <= eps0:
        return ((integrate.quad(lambda x: 1.0 / 2.0, 0, 3 * eps)[0] +
                 integrate.quad(f1, 3.0 * eps, eps / eps0, eps)[0] +
                 integrate.quad(f2, eps / eps0, 1.0, eps)[0]
                 ))


def trevisan_function(eps):

    if eps <= 1.0 / 16.0:
        return 1.0 - 4.0 * np.sqrt(eps) + 8.0 * eps
    return 1.0 / 2.0


#===============================================================================
# Greedy Cut
#
# Approximation ratio: 0.5
# Complexity: O(n^2)
#===============================================================================


def greedy_choice(G, candidate, blue_nodes, black_nodes, visited):
    """Helper function to greedy cut"""

    G.node[candidate][gc.PARTITION] = gc.BLUE
    blue_cut_val = gc.cut_edges(nx.subgraph(G, visited))

    G.node[candidate][gc.PARTITION] = gc.BLACK
    black_cut_val = gc.cut_edges(nx.subgraph(G, visited))

    if blue_cut_val > black_cut_val:
        G.node[candidate][gc.PARTITION] = gc.BLUE
        blue_nodes.add(candidate)
    else:
        black_nodes.add(candidate)

    return blue_nodes, black_nodes


def greedy_cut(G, nbunch=None, visited=None):
    """Return a good cut of a graph G.

    Good cut: a cut is good if it cuts at least half of the number of
    edges of the graph.

    """
    if nbunch == None:
        nbunch = G.nodes()  # set of nbunch

    if visited is None:
        visited = set()

    gc.init_cut(G, nbunch)

    blue_nodes = set()
    black_nodes = set()

    candidate = nbunch.pop()
    visited.add(candidate)

    greedy_choice(G, candidate, blue_nodes, black_nodes, visited)

    while nbunch:

        candidate = nbunch.pop()
        visited.add(candidate)

        greedy_choice(G, candidate, blue_nodes, black_nodes, visited)

    gc.set_partitions(G, blue_nodes, black_nodes)
    return blue_nodes, black_nodes


#==============================================================================
# 2TSC Approximation Algorithm by Luca Trevisan
#
# Approximation ratio: 0.531
# Complexity: O(n^2)
#==============================================================================


def first_lemma(G, y):
    """Compute first lemma."""

    numerator = 0.0
    for i in y:
        for j in y:
            if G.has_edge(i, j):
                numerator += abs(y[i] + y[j])

    denominator = 0.0
    for i in y:
        denominator += G.degree(i) * abs(y[i])

    # float division by zero
    if denominator == 0:
        return None
    return numerator / denominator


def second_lemma(G):
    """Compute second lemma."""
    cut_edges = gc.cut_edges(G)
    uncut_edges = G.number_of_edges() - cut_edges

    numerator = float(uncut_edges - cut_edges)
    denominator = float(G.number_of_edges())

    return numerator / denominator


def largest_eigenvector(G):
    """Return the largest eigenvector of a graph G."""
    L = nx.normalized_laplacian_matrix(G)

    eigenvalues, eigenvectors = np.linalg.eig(L)

    # highest eigenvalue index and ...
    ind = np.argmax(eigenvalues)
    # ... its corresponding eigenvector.
    largest = eigenvectors[:, ind]

    return dict(zip(G, largest))


def two_threshold_spectral_cut(G):
    """Return an indicator vector of a cut computed using the largest
    eigenvector.

    """
    x = largest_eigenvector(G)

    smallest = gc.sign_norm(dict(x))  # all 1 and -1 vector
    min_ratio = first_lemma(G, smallest)

    y = dict.fromkeys(x, 0)

    for k in x:

        for i in x:

            if x[i] < -abs(x[k]):
                y[i] = -1
            elif x[i] > abs(x[k]):
                y[i] = 1
            elif abs(x[i]) <= abs(x[k]):
                y[i] = 0

        # compute first lemma
        ratio = first_lemma(G, y)

        if ratio is not None:
            if min_ratio is None:
                min_ratio, smallest = ratio, dict(y)
            elif ratio < min_ratio:
                min_ratio, smallest = ratio, dict(y)

    # smallest ratio's vector
    return smallest


def recursive_spectral_cut(G):
    """Return an approximate solution to the max cut problem.

    Use the two_threshold_spectral_cut and recursively cut
    the undecided nodes.

    """

    if not G or G.number_of_nodes() == 0:
        return set(), set()

    smallest = two_threshold_spectral_cut(G)

    R = set()
    L = set()
    V = set()

    for i in smallest:
        if smallest[i] == 1:  # blue
            R.add(i)
        elif smallest[i] == 0:  # magenta
            V.add(i)
        elif smallest[i] == -1:  # black
            L.add(i)

    G1 = nx.Graph(nx.subgraph(G, V))

    M = G.number_of_edges() - G1.number_of_edges()
    C = gc.edges_beetween(G, L, R)  # cut edges
    X = gc.edges_beetween(G, L, V) + gc.edges_beetween(G, R, V)

    if C + 0.5 * X <= 0.5 * M or not V:
        if gc.edges_beetween(G, L, R) < G.number_of_edges() / 2:
            return greedy_cut(G)
        return L, R

    if C + 0.5 * X > 0.5 * M:

        # SPECIAL CASE: all undecided nodes (V) are isolate deg = 0
        if gc.is_all_isolate(G1):
            gc.set_partitions(G, L, R)
            visited = (L | R) - V
            B, K = greedy_cut(G, V, visited)
            return L | B, R | K

        V1, V2 = recursive_spectral_cut(G1)

        if (gc.edges_beetween(G, V1 | L, V2 | R) > 
            gc.edges_beetween(G, V1 | R, V2 | L)):
            return V1 | L, V2 | R
        return V1 | R, V2 | L


def trevisan_approximation_alg(G):

    B, K = recursive_spectral_cut(G)
    # set blue and black nodes in graph G
    gc.set_partitions(G, B, K)
    return gc.cut_edges(G)


#==============================================================================
# Enumerative Methods for Maximum Cut's Exact Solutions by Andrea Casini
#
# Complexity: O(2^n)
#==============================================================================


def brute_force_max_cut(G):
    """Compute maximum cut of a graph considering all the possible cuts."""

    max_cut_value = 0
    max_cut_ind = 0

    n = G.number_of_nodes()

    for i in range(1, 2 ** (n - 1)):
        cut_graph = nx.Graph(G)

        gc.binary_cut(cut_graph, i)
        value = gc.cut_edges(cut_graph)

        if value > max_cut_value:
            max_cut_value = value
            max_cut_ind = i

    gc.binary_cut(G, max_cut_ind)
    return gc.partition_dictionary(G), max_cut_value


#===============================================================================
# Fast Max Cut
#===============================================================================


def choose_new_candidate(partition_dict, nodes_stack):
    """Return the first candidate node.

    Candidate node: first undecided at top of the stack.

    """
    candidate = nodes_stack.pop()

    # choose the first marked or undecided node
    while(partition_dict[candidate] != gc.UNDECIDED and
          partition_dict[candidate] != gc.MARKED and
          nodes_stack):

        candidate = nodes_stack.pop()

    return candidate


def aux_local_consistent_max_cut(G,
                                 partition_dict,
                                 degree_node_seq,
                                 candidate,
                                 partition_attribute):
    """Helper function to 'local_consistent_max_cut'."""
    partition_dict[candidate] = partition_attribute

    if not degree_node_seq:

        if not gc.is_cut_consistent(G, partition_dict):
            return None, 0

        return partition_dict, gc.cut_edges(G, partition_dict)

    while(gc.could_be_cut(G, partition_dict)):
        pass

    # pick a new candidate
    candidate = choose_new_candidate(partition_dict, degree_node_seq)

    blue_cut, blue_cut_val = aux_local_consistent_max_cut(
                                                  G,
                                                  dict(partition_dict),
                                                  list(degree_node_seq),
                                                  candidate,
                                                  gc.BLUE)

    black_cut, black_cut_val = aux_local_consistent_max_cut(
                                                     G,
                                                     dict(partition_dict),
                                                     list(degree_node_seq),
                                                     candidate,
                                                     gc.BLACK)

    if blue_cut is None and black_cut is None:
        return None, 0

    # Choose best cut according to the effective cut value
    if blue_cut_val > black_cut_val:
        return blue_cut, blue_cut_val
    return black_cut, black_cut_val


def aux_pruning_local_consistent_max_cut(G,
                                         partition_dict,
                                         degree_node_seq,
                                         candidate,
                                         partition_attribute):
    """Helper function to pruning_local_consistent_max_cut"""
    partition_dict[candidate] = partition_attribute

    # check consistency in the candidate and its neighbors only
    check_nodes = nx.neighbors(G, candidate)

    if not gc.is_cut_consistent(G, partition_dict, check_nodes):
        return None, 0

    if not degree_node_seq:
        return partition_dict, gc.cut_edges(G, partition_dict)

    while(gc.could_be_cut(G, partition_dict)):
        pass

    # pick a new candidate
    candidate = choose_new_candidate(partition_dict, degree_node_seq)

    blue_cut, blue_cut_val = aux_pruning_local_consistent_max_cut(
                                                          G,
                                                          dict(partition_dict),
                                                          list(degree_node_seq),
                                                          candidate,
                                                          gc.BLUE)

    black_cut, black_cut_val = aux_pruning_local_consistent_max_cut(
                                                            G,
                                                            dict(partition_dict),
                                                            list(degree_node_seq),
                                                            candidate,
                                                            gc.BLACK)

    if blue_cut is None and black_cut is None:
        return None, 0

    # Choose best cut according to the effective cut value
    if blue_cut_val > black_cut_val:
        return blue_cut, blue_cut_val
    return black_cut, black_cut_val


def local_consistent_max_cut(G, lowest=False, pruning=False):
    """Compute maximum cut of a graph taking advantage of
    the consistency property.

    """
    partition_dict = dict.fromkeys(G, gc.UNDECIDED)  # build a cut dictionary
    deg_node_seq = gc.degree_nodes_sequence(G, reverse=lowest)

    candidate = choose_new_candidate(partition_dict, deg_node_seq)

    if not pruning:
        max_cut_dict, max_cut_value = aux_local_consistent_max_cut(
                                                            G,
                                                            partition_dict,
                                                            deg_node_seq,
                                                            candidate,
                                                            gc.BLUE)
    else:
        max_cut_dict, max_cut_value = aux_pruning_local_consistent_max_cut(
                                                                   G,
                                                                   partition_dict,
                                                                   deg_node_seq,
                                                                   candidate,
                                                                   gc.BLUE)

    gc.cut(G, max_cut_dict)
    return max_cut_dict, max_cut_value


#==============================================================================
# Faster Max Cut
#==============================================================================


def compute_estimated_cut(G, partition_dict, marked_nodes):
    """Compute overestimated cut value on a partial marked graph"""

    buffer_dict = dict(partition_dict)

    # cut marked nodes using strong minority
    gc.marked_nodes_could_be_cut(G, buffer_dict, marked_nodes)

    bk = set()  # decided nodes in partition_dict
    m = set()  # marked nodes in partition_dict

    for node in partition_dict:
        if partition_dict[node] == gc.MARKED:
            m.add(node)
        else:
            bk.add(node)

    b1k1 = set()  # decided and undecided nodes in buffer_dict

    for node in m:
        if buffer_dict[node] != gc.MARKED:
            b1k1.add(node)

    nx.set_node_attributes(G, gc.PARTITION, buffer_dict)

    result = (G.subgraph(m).number_of_edges() +
              gc.cut_edges(G.subgraph(bk | b1k1)) -
              gc.cut_edges(G.subgraph(b1k1)))

    return result


def aux_lazy_local_consistent_max_cut(G,
                                      partition_dict,
                                      nodes_stack,
                                      candidate,
                                      partition_attribute,
                                      marked_nodes,
                                      consistent_cuts):
    """Helper function. 
    Computes an estimated maximum cut based on an overestimated
    cut value.

    """
    partition_dict[candidate] = partition_attribute

    if not nodes_stack:
    
        if not gc.is_cut_consistent(G, partition_dict):
            return (None, 0)

        estimated_cut_val = compute_estimated_cut(G,
                                                  partition_dict,
                                                  marked_nodes)

        consistent_cuts.append((partition_dict, estimated_cut_val))
        return partition_dict, estimated_cut_val

    while(gc.could_be_cut(G, partition_dict)):
        pass

    candidate = choose_new_candidate(partition_dict, nodes_stack)

    blue_cut, blue_cut_val = aux_lazy_local_consistent_max_cut(
                                               G,
                                               dict(partition_dict),
                                               list(nodes_stack),
                                               candidate,
                                               gc.BLUE,
                                               marked_nodes,
                                               consistent_cuts)

    black_cut, black_cut_val =  aux_lazy_local_consistent_max_cut(
                                                   G,
                                                   dict(partition_dict),
                                                   list(nodes_stack),
                                                   candidate,
                                                   gc.BLACK,
                                                   marked_nodes,
                                                   consistent_cuts)

    if blue_cut is None and black_cut is None:
        return None, 0

    # Choose best cut according to the overestimated cut value
    if blue_cut_val > black_cut_val:
        return blue_cut, blue_cut_val
    return black_cut, black_cut_val


def complete_cut(G, partition_dict, nodes_stack):
    """Compute the maximum cut from a partial partitioned graph."""

    marked_nodes_stack = []

    while gc.could_be_cut(G, partition_dict):
        pass

    # filter away the decided nodes
    for i in nodes_stack:
        if partition_dict[i] is gc.UNDECIDED or partition_dict[i] is gc.MARKED:
            marked_nodes_stack.append(i)

    if not marked_nodes_stack:
        return partition_dict, gc.cut_edges(G, partition_dict)

    candidate = marked_nodes_stack.pop()

    # Complete cut
    blue_cut, blue_cut_val = aux_local_consistent_max_cut(
                                                  G,
                                                  dict(partition_dict),
                                                  list(marked_nodes_stack),
                                                  int(candidate),
                                                  gc.BLUE)

    black_cut, black_cut_val = aux_local_consistent_max_cut(
                                                    G,
                                                    dict(partition_dict),
                                                    list(marked_nodes_stack),
                                                    int(candidate),
                                                    gc.BLACK)

    if blue_cut_val > black_cut_val:
        return blue_cut, blue_cut_val
    return black_cut, black_cut_val


def do_work(G,
            work_queue,
            max_cut_dict,
            max_cut_value,
            max_cuts_queue,
            marked_deg_nodes):
    """Work function dedicated for multiprocessing only. DO NOT USE"""
    consistent_cuts = work_queue.get()

    for cons_cut_dict, cons_cut_value in consistent_cuts:

        a_cut_dict, a_cut_value = complete_cut(G,
                                                  cons_cut_dict,
                                                  list(marked_deg_nodes))

        if a_cut_value > max_cut_value:
            max_cut_value = a_cut_value
            max_cut_dict = a_cut_dict

    max_cuts_queue.put((max_cut_dict, max_cut_value))


@timeout(900)  # 15 minutes
def lazy_local_consistent_max_cut(G, marked_nodes=None, strategy=0, parallel=False):
    """Compute the maximum cut of a graph using the consistency property
    and a marking nodes strategy (0:best).

    """
    partition_dict = {}
    consistent_cuts = []

    # Choose the marked nodes according to the chosen marking strategy
    if marked_nodes is None:

        if strategy == 0:  # apparently the best
            marked_nodes = gc.lowest_degree_nodes(G, G.number_of_nodes() / 2)

        elif strategy == 1:
            marked_nodes = gc.two_maximal_independent_set(G)

        elif strategy == 2:
            marked_nodes = gc.pick_random_nodes(G, G.number_of_nodes()  / 2)

        elif strategy == 3:  # do not use never terminate
            marked_nodes = gc.highest_degree_nodes(G, G.number_of_nodes() / 2)

    # Initialize the partition dictionary: marked nodes and undecided nodes
    for i in G.nodes():
        if i in marked_nodes:
            partition_dict[i] = gc.MARKED
        else:
            partition_dict[i] = gc.UNDECIDED

    # Nodes list in ascending order according to the node's degree
    deg_node_seq = gc.degree_nodes_sequence(G)

    unmarked_deg_nodes = []
    marked_deg_nodes = []

    for node in deg_node_seq:
        if node in marked_nodes:
            marked_deg_nodes.append(node)  # undecided nodes will be here
        else:
            unmarked_deg_nodes.append(node)

    # Choose candidate from the unmarked nodes
    candidate = unmarked_deg_nodes.pop()

    # Compute highest overestimated cut value over the unmarked nodes
    estimated_max_cut = aux_lazy_local_consistent_max_cut(
                                          G,
                                          dict(partition_dict),
                                          unmarked_deg_nodes,
                                          candidate,
                                          gc.BLUE,
                                          marked_nodes,
                                          consistent_cuts)[0]

    # Compute the effective cut based on the estimated one
    max_cut_dict, effective_cut_value = complete_cut(G,
                                                     estimated_max_cut,
                                                     list(marked_deg_nodes))

    filtered_cons_cuts = []

    # filter the consistent cuts
    for cons_cut_dict, cons_cut_value in consistent_cuts:
        if cons_cut_value > effective_cut_value:
            filtered_cons_cuts.append((cons_cut_dict, cons_cut_value))

    max_cut_value = 0

    # if the work load is too low than there's no speed up with multiprocessing
    if len(filtered_cons_cuts) < 10:
        parallel = False

    if not parallel:

        for cons_cut_dict, cons_cut_value in filtered_cons_cuts:

            a_cut_dict, a_cut_value = complete_cut(G,
                                                   cons_cut_dict,
                                                   list(marked_deg_nodes))

            if a_cut_value > max_cut_value:
                max_cut_value = a_cut_value
                max_cut_dict = a_cut_dict

    else:

        cpus = cpu_count()  # Number of cores
        size = len(filtered_cons_cuts) / cpus + 1  # Work load for each process

        work_queue = Queue()  # Each element is a work for a single process
        max_cuts_queue = Queue()  # where to put results

        # Distribute the work in a queue
        for i in range(cpus):
            j = size * i
            work_queue.put(filtered_cons_cuts[j: j + size])

        # Start a number of process equivalent to the number of core installed
        processes = [Process(target=do_work,
                             args=(G,
                                   work_queue,
                                   max_cut_dict,
                                   max_cut_value,
                                   max_cuts_queue,
                                   marked_deg_nodes))

                     for i in range(cpus)]

        # handle the time out exception
        try:

            for p in processes:
                p.start()

            for p in processes:
                p.join()

        except TimedOutExc:  # kill processes

            for p in processes:
                p.terminate()
            return None

        # Iterate over the queue and get the maximum cut
        while not max_cuts_queue.empty():
            a_cut_dict, a_cut_value = max_cuts_queue.get()

            if a_cut_value > max_cut_value:
                max_cut_value = a_cut_value
                max_cut_dict = a_cut_dict

    gc.cut(G, max_cut_dict)
    return max_cut_dict, max_cut_value

