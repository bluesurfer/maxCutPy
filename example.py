"""
Just a usage example
"""

import networkx as nx
import matplotlib.pylab as plt

import maxcutpy.graphdraw as gd
import maxcutpy.maxcut as mc
import maxcutpy.graphcut as gc
import maxcutpy.graphtest as gt


__license__ = "GPL"


if __name__ == '__main__':

    seed = 123

    # most used graphs
    G1 = nx.erdos_renyi_graph(n=24, p=0.3, seed=seed)

    # some cool graphs
    G2 = nx.star_graph(20)
    G3 = nx.path_graph(30)
    G4 = nx.petersen_graph()
    G5 = nx.dodecahedral_graph()
    G6 = nx.house_graph()
    G7 = nx.moebius_kantor_graph()
    G8 = nx.barabasi_albert_graph(5, 4)
    G9 = nx.heawood_graph()
    G10 = nx.icosahedral_graph()
    G11 = nx.sedgewick_maze_graph()
    G12 = nx.havel_hakimi_graph([1, 1])
    G13 = nx.complete_graph(20)
    G14 = nx.bull_graph()

    G = G1  # choose a graph from the list

    gd.draw_custom(G)
    plt.show()

    #exact cut
    print("Time 'local_consistent_max_cut':" + str(gt.execution_time(mc.local_consistent_max_cut, 1, G)))
    print('Edges cut: ' + str(gc.cut_edges(G)))
    print('\n')
    print("Time 'lazy_local_consistent_max_cut':" + str(gt.execution_time(mc.lazy_local_consistent_max_cut, 1, G)))
    print('Edges cut: ' + str(gc.cut_edges(G)))
    print('\n')

    gd.draw_cut_graph(G)
    plt.show()

    #approximated cut
    print("Time 'trevisan_approximation_alg': " + str(gt.execution_time(mc.trevisan_approximation_alg, 1, G)))
    print('Edges cut: ' + str(gc.cut_edges(G)))

    gd.draw_cut_graph(G)
    plt.show()

    print('\n')
    print('Time Greedy: ' + str(gt.execution_time(mc.greedy_cut, 1, G)))
    print('Edges cut: ' + str(gc.cut_edges(G)))

    gd.draw_cut_graph(G)
    plt.show()


