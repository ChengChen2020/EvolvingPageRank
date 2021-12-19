import re
import os
import random
# import numpy as np
# import matplotlib.pyplot as plt

from structure import Graph


def parse_nodes_constant(all_edges, nodes):
    # Only get edges contain nodes

    filtered_all_edges = []

    for edges in all_edges:
        filtered_edges = []
        for e in edges:
            FromNodeId, ToNodeId = re.findall('[0-9]+', e.strip())[:2]
            if FromNodeId in nodes and ToNodeId in nodes:
                filtered_edges.append(e)
        filtered_all_edges.append(filtered_edges)

    return filtered_all_edges


def get_nodes(edges):
    # Get common nodes
    nodes = set()

    for e in edges:
        FromNodeId, ToNodeId = re.findall('[0-9]+', e.strip())[:2]
        nodes.add(FromNodeId)
        nodes.add(ToNodeId)

    return nodes


def parse_as(file_root, file_names, skip_lines=2, num=100, thresh=500):
    raw_evo_nodes = []
    raw_evo_edges = []
    raw_evo_lines = []

    for graph_name in sorted(file_names)[:num]:

        with open(os.path.join(file_root, graph_name)) as g:
            for _ in range(skip_lines):
                g.readline()

            n, e = re.findall('[0-9]+', g.readline().strip())

            raw_evo_nodes.append(int(n))
            raw_evo_edges.append(int(e))

            g.readline()
            lines = g.readlines()

            raw_evo_lines.append(lines)

    evo_nodes = []
    evo_edges = []
    evo_lines = []

    # Prune abrupt changes
    # Smooth the sequence
    prev = raw_evo_nodes[0]
    for i in range(1, len(raw_evo_nodes)):
        if abs(raw_evo_nodes[i] - prev) < thresh:
            evo_nodes.append(raw_evo_nodes[i])
            evo_edges.append(raw_evo_edges[i])
            evo_lines.append(raw_evo_lines[i])
            prev = raw_evo_nodes[i]

    # plt.figure(figsize=(18, 6))
    # plt.plot(np.arange(len(evo_nodes)), evo_nodes, '*-', color='g', label="nodes")
    # # plt.plot(np.arange(num), evo_edges, label="edges")
    # plt.locator_params(axis="y", nbins=10)
    # plt.legend(loc="best")
    # plt.savefig(file_root.split('/')[-1])

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
        # Random Probing
        return random.choices(graph.nodes, k=num)
    elif strategy == 'rr':
        # Round-Robin Probing
        return graph.return_rr_k(num)
    elif strategy == 'wr':
        # Proportional Probing
        PageRank(graph, 0.15, 500)
        return random.choices(graph.nodes, weights=graph.get_pagerank_list(), k=num)
    else:
        # Priority Probing
        return graph.return_top_k(num)


def evolve_graph(evo_graph, target_graph, strategy, num, damping_factor, iterations):
    # Probing nodes from evolving graph
    probing_nodes = get_nodes_to_probe(evo_graph, strategy, num)
    assert len(probing_nodes) == num

    for pn in probing_nodes:

        if pn not in evo_graph.nodes:
            continue

        if pn.name in target_graph.node_names:
            evo_graph.add_node(pn, target_graph)
        else:
            evo_graph.delete_node(pn)

    PageRank(evo_graph, damping_factor, iterations)

    # Priority
    if strategy == 'pr':
        for node in evo_graph.nodes:
            if node in probing_nodes:
                node.priority = 0.0
            else:
                node.priority += node.pagerank


def PageRank_one_iter(graph, d):
    node_list = graph.nodes
    for node in node_list:
        node.update_pagerank(d, len(graph.nodes))
    graph.update_pagerank()
    graph.normalize_pagerank()


def PageRank(graph, d, iteration=100):
    graph.clear_pagerank()
    graph.prune()
    for i in range(iteration):
        PageRank_one_iter(graph, d)


def l1_error(evo_graph, true_graph):
    inter_nodes = list(set(evo_graph.node_names).intersection(set(true_graph.node_names)))
    import statistics
    return statistics.mean([abs(evo_graph.find(name).pagerank - true_graph.find(name).pagerank) for name in inter_nodes])