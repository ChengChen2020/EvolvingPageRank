import re
import os
import random
from copy import deepcopy

from structure import Graph


def parse_as(file_root, file_names, skip_lines=2):
    evo_nodes = []
    evo_edges = []
    evo_lines = []
    for graph_name in sorted(file_names):

        with open(os.path.join(file_root, graph_name)) as g:
            for _ in range(skip_lines):
                g.readline()

            n, e = re.findall('[0-9]+', g.readline().strip())

            evo_nodes.append(int(n))
            evo_edges.append(int(e))

            g.readline()
            lines = g.readlines()

            evo_lines.append(lines)

    #     plt.figure(figsize=(12, 6))
    #     plt.plot(np.arange(len(file_names)), evo_nodes, label="nodes")
    #     plt.plot(np.arange(len(file_names)), evo_edges, label="edges")
    #     plt.locator_params(axis="y", nbins=10)
    #     plt.legend(loc="best")

    return evo_nodes, evo_edges, evo_lines


def build_graph(lines, name='as-caida'):
    graph = Graph(name)

    for line in lines:
        FromNodeId, ToNodeId = re.findall('[0-9]+', line.strip())[:2]
        graph.add_edge(FromNodeId, ToNodeId)

    random.shuffle(graph.nodes)

    return graph


def get_nodes_to_probe(graph, strategy, num=10):
    if strategy not in ['rd', 'rr', 'wr', 'pr']:
        print("No such strategy")
        assert False
    if strategy == 'rd':
        # Random
        return random.choices(graph.nodes, k=num)
    elif strategy == 'rr':
        # Round Robin
        return graph.return_rr_k(num)
    elif strategy == 'wr':
        # Weighted Random
        # PageRank(graph, damping_factor, iterations)
        return random.choices(graph.nodes, weights=graph.get_pagerank_list(), k=num)
    else:
        # Priority Probing
        # PageRank(graph, damping_factor, iterations)
        return graph.return_top_k(num)


def evolve_graph(evo_graph, target_graph, strategy):
    # probing nodes from evolving graph
    probing_nodes = get_nodes_to_probe(evo_graph, strategy)

    # for node in probing_nodes:
    #     print(node.pagerank)

    for pn in probing_nodes:

        assert pn in evo_graph.nodes

        if pn.name in target_graph.node_names:
            evo_graph.add_node(target_graph.find(pn.name))
        else:
            evo_graph.delete_node(pn)

    evo_graph.clear_pagerank()


def PageRank_one_iter(graph, d):
    node_list = graph.nodes
    for node in node_list:
        node.update_pagerank(d, len(graph.nodes))
    graph.normalize_pagerank()


def PageRank(graph, d, iteration=100):
    for i in range(iteration):
        PageRank_one_iter(graph, d)


def l1_error(evo_graph, true_graph):
    inter_nodes = list(set(evo_graph.node_names).intersection(set(true_graph.node_names)))
    return sum([abs(evo_graph.find(name).pagerank - true_graph.find(name).pagerank) for name in inter_nodes])
