"""
Graphic library with pretty drawing methods for graphs using matplotlib.

"""

import networkx as nx
import matplotlib.pylab as plt

import graphcut as gc


__license__ = "GPL"


def draw_custom(G, pos=None,
                node_size=1000,
                edge_width=3,
                font_size=12,
                node_color='white',
                color_map='Blues',
                edge_label=None):
    """Draw the graph G using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.

    """
    if not pos:
        pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color,
                           node_size=node_size
    )

    nx.draw_networkx_edges(G, pos,
                           width=edge_width,
                           edge_color=range(nx.number_of_edges(G)),
                           edge_cmap=plt.get_cmap(color_map),
                           edge_vmin=-10,
                           edge_vmax=10)

    nx.draw_networkx_labels(G,
                            pos,
                            font_size=font_size)

    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=nx.get_edge_attributes(G, edge_label))

    plt.axis('off')


def draw_graphs_list(graphs_list, node_size=100, edge_width=2):
    """Draw a list of graphs using Matplotlib and Pygraphviz."""

    a_graph = nx.Graph()

    for graph in graphs_list:
        a_graph = nx.disjoint_union(a_graph, graph)

    pos = nx.graphviz_layout(a_graph, prog='neato')

    C = nx.connected_component_subgraphs(a_graph)

    for c in C:
        draw_cut_graph(c,
                       pos=pos,
                       node_size=node_size,
                       edge_width=edge_width,
                       node_label=False)


def draw_cut_graph(G,
                   partition_dict=None,
                   pos=None,
                   node_size=1000,
                   edge_width=3,
                   font_size=12,
                   node_label=True,
                   title=''):
    """Draw a cut graph G using Matplotlib."""

    if partition_dict:
        nx.set_node_attributes(G, gc.PARTITION, partition_dict)

    if not pos:
        pos = nx.circular_layout(G, scale=20)

    blue_nodes, black_nodes, undecided_nodes, marked_nodes = gc.get_partitions(G)

    # Draw nodes and edges of the first partition

    nx.draw_networkx_nodes(G, pos,
                           blue_nodes,
                           node_size=node_size,
                           node_color='blue')

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, blue_nodes, blue_nodes),
                           width=edge_width,
                           edge_color='blue')

    # Draw nodes and edges of the second partition

    nx.draw_networkx_nodes(G, pos,
                           black_nodes,
                           node_size=node_size,
                           node_color='black')

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, black_nodes, black_nodes),
                           width=edge_width,
                           edge_color='black')

    # Draw undecided nodes and edges

    nx.draw_networkx_nodes(G, pos,
                           undecided_nodes,
                           node_size=node_size,
                           node_color='magenta')

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, undecided_nodes, undecided_nodes),
                           width=edge_width,
                           edge_color='magenta')

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, blue_nodes, undecided_nodes),
                           width=edge_width,
                           style='dotted',
                           edge_color='magenta')

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, undecided_nodes, black_nodes),
                           width=edge_width,
                           style='dotted',
                           edge_color='magenta')

    # Draw marked nodes and edges

    nx.draw_networkx_nodes(G, pos,
                           marked_nodes,
                           node_size=node_size,
                           node_color='red')

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, marked_nodes, marked_nodes),
                           width=edge_width,
                           edge_color='red')

    #Draw edges beetween marked and unmarked

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, marked_nodes, blue_nodes),
                           width=edge_width,
                           edge_color='orange')

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, marked_nodes, black_nodes),
                           width=edge_width,
                           edge_color='orange')

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, marked_nodes, undecided_nodes),
                           width=edge_width,
                           edge_color='orange')

    # Draw cut edges

    nx.draw_networkx_edges(G, pos,
                           nx.edge_boundary(G, blue_nodes, black_nodes),
                           width=edge_width,
                           style='dashed',
                           edge_color='gray')

    if node_label:
        nx.draw_networkx_labels(G,
                                pos,
                                font_color='white',
                                font_size=font_size,
                                font_weight='bold')

    plt.title(title)
    plt.axis('off')


