"""
Testing library. Contains some methods for testing cut algorithm

"""

import time
from timeit import Timer
from functools import partial
from collections import defaultdict

import networkx as nx

import graphcut as gc
import graphgen as gg


__license__ = "GPL"


def execution_time(function, n_times=1, *args, **kwargs):
    """Return the execution time of a function in seconds."""
    return round(Timer(partial(function, *args, **kwargs))
                 .timeit(n_times), 3)


def execution_time_and_output(function, n_times=1, *args, **kwargs):
    """Return the execution time of a function in seconds together with
    its output."""
    start = time.time()
    retval = function(*args, **kwargs)
    elapsed = time.time() - start
    return round(elapsed, 3), retval


#==============================================================================
# Tests
#==============================================================================


def test_cut_algorithm(cut_alg,
                       benchmark_filename,
                       test_name='',
                       n_times=1,
                       *args,
                       **kargs):
    """Test cut algorithm time performance and epsilon values.

    Parameters
    ----------
    cut_alg: the exact cut algorithm you want to test

    benchmark_filename: a dictionary of adjacency matrices lists indexed by a 
                        key value.

    """
    times = []
    epsilons = []
    keys = []
    total_time = 0.0

    adj_matrix_dict = gg.read_from_file(benchmark_filename + '.dat', 
                                        gg.BENCHMARKS_FILE_DIR)

    epsilons_dict = defaultdict(list)

    print('Testing: ' + str(cut_alg.__name__))
    print('Benchmark: ' + benchmark_filename)
    print('Starting..:)')
    print('Graph\tKey\tEpsilon\t\tExecution Time')

    i = 1  # iteration counter

    for key in sorted(adj_matrix_dict.iterkeys()):
        G = nx.from_numpy_matrix(adj_matrix_dict[key])
        t, retval = execution_time_and_output(cut_alg,
                                              n_times,
                                              G,
                                              *args,
                                              **kargs)

        if retval == None:
            print('Time out! ' + str(t))
            continue

        e = gc.compute_epsilon(G)

        print(str(i) + '\t' + str(round(key, 3)) + '\t' + str(e) + '\t\t' + str(t))

        keys.append(key)
        times.append(t)
        epsilons.append(e)
        total_time += t
        i += 1

        epsilons_dict[e].append((nx.adj_matrix(G), gc.partition_dictionary(G)))
        gg.write_to_file(epsilons_dict, 'graphs_' + benchmark_filename + '.dat')

    gg.write_to_file((keys, times, epsilons),
                      test_name + benchmark_filename + '.dat',
                      cut_alg.__name__)

    print('Total time:\t\t\t' + str(total_time))
    print('Success\n')


def compare_cut_algorithm_results(cut_alg, results_graphs, test_name=''):

    graphs_dict = gg.read_from_file(results_graphs + '.dat')
    location = cut_alg.__name__

    error_list = []
    results_dict = defaultdict(list)

    print('Algorithm: ' + str(cut_alg.__name__))
    print('Benchamark: ' + results_graphs)
    print('Starting..:)')
    print('Graph\Result Epsilon\tCurrent Epsilon')

    i = 0

    for opt_eps in sorted(graphs_dict.iterkeys()):
        for A, c in graphs_dict[opt_eps]:

            G = nx.from_numpy_matrix(A)
            cut_alg(G)

            if gc.cut_edges(G) < G.number_of_edges() / 2:
                print('####### ERROR 1 #######')
                error_list.append(G)

            if gc.are_undecided_nodes(G):
                print('####### ERROR 2 #######')
                error_list.append(G)

            this_eps = gc.compute_epsilon(G)
            print(str(i) + '\t' + str(opt_eps) + '\t' + str(this_eps))
            results_dict[opt_eps].append(this_eps)

            i += 1

    gg.write_to_file(results_dict, test_name + 'results_' + results_graphs + '.dat', location)

    if len(error_list) != 0:
        gg.write_to_file(error_list, test_name + 'errors_' + results_graphs + '.dat', location)
