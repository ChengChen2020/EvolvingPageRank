import sys
import time
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import *

sys.setrecursionlimit(10000)

# https://github.com/chonyy/PageRank-HITS-SimRank

parser = argparse.ArgumentParser()
parser.add_argument('--damping factor', type=float, default=0.15, help='damping factor')
parser.add_argument('--iterations', type=int, default=500, help='iterations')
parser.add_argument('--probing_nodes_num', type=int, default=10, help='number of probing nodes')
parser.add_argument('--fig_path', type=str, default='analysis.png', help='learning rate')
opt = parser.parse_args()

# undirected
# with loop
dsroot_1 = "../data/as-733/"
# directed
# no loop
dsroot_2 = "../data/as-caida/"

graphnames_1 = sorted([f for f in os.listdir(dsroot_1) if os.path.isfile(os.path.join(dsroot_1, f))])
graphnames_2 = sorted([f for f in os.listdir(dsroot_2) if os.path.isfile(os.path.join(dsroot_2, f))])

assert (len(graphnames_1) == 733)
assert (len(graphnames_2) == 122)

damping_factor = 0.15
iterations = 500
probing_nodes_num = 10  # propotional to current number of nodes


def test():
    # t = Graph()
    # t.add_edge('D', 'A')
    # t.add_edge('D', 'B')
    # t.add_edge('C', 'B')
    # t.add_edge('B', 'C')
    # t.add_edge('F', 'E')
    # t.add_edge('E', 'F')
    # t.add_edge('E', 'D')
    # t.add_edge('E', 'B')
    # t.add_edge('F', 'B')
    # t.add_edge('G', 'B')
    # t.add_edge('G', 'E')
    # t.add_edge('H', 'B')
    # t.add_edge('H', 'E')
    # t.add_edge('I', 'B')
    # t.add_edge('I', 'E')
    # t.add_edge('J', 'E')
    # t.add_edge('K', 'E')
    # # t.display()
    # PageRank(t, damping_factor, iterations)
    # print(t.get_pagerank_list())

    # h = Graph()
    # h.add_edge(1, 3)
    # h.add_edge(3, 1)
    # h.add_edge(3, 2)
    # h.add_edge(2, 3)
    # h.add_edge(3, 4)
    # h.add_edge(1, 4)
    # h.add_edge(4, 2)
    # PageRank(h, damping_factor, 10)
    # print(h.get_pagerank_list())

    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    random.shuffle(g.nodes)

    PageRank(g, damping_factor, iterations)
    print(g.get_pagerank_list())

    g2 = Graph()
    g2.add_edge(1, 2)
    g2.add_edge(2, 3)
    g2.add_edge(2, 4)
    g2.add_edge(4, 3)
    g2.add_edge(4, 3)
    g2.add_edge(4, 1)
    g2.add_edge(3, 5)
    random.shuffle(g2.nodes)

    evolve_graph(g2, g, 'rr', 2, opt.damping_factor, opt.iterations)

    g2.display()

    PageRank(g, damping_factor, iterations)
    print(g.get_pagerank_list())
    PageRank(g2, damping_factor, iterations)
    print(g2.get_pagerank_list())

    print(l1_error(g, g2))


def main():
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Parse CAIDA
    # caida_nodes, caida_edges, caida_lines = parse_as(dsroot_2, graphnames_2, skip_lines=6)
    # Parse as-733
    start = time.time()
    print('Parsing as-733')
    as_nodes, as_edges, as_lines = parse_as(dsroot_1, graphnames_1, skip_lines=2)
    print('Parsing as-733 done')

    graphs = []
    for i in tqdm(range(1, 10)):
        graph = build_graph(as_lines[i], 'as-733')
        PageRank(graph, damping_factor, iterations)
        graphs.append(build_graph(as_lines[i], 'as-733'))
    print('Building graphs: ', time.time() - start)

    err = defaultdict(list)
    color = {'rd': 'b', 'rr': 'r', 'wr': 'g', 'pr': 'y'}

    for stgy in ['rd', 'rr', 'wr', 'pr']:

        evo_graph = build_graph(as_lines[0], 'as-733')
        PageRank(evo_graph, damping_factor, iterations)

        for graph in graphs:
            evolve_graph(evo_graph, graph, stgy, 100, damping_factor, iterations)
            print(stgy, len(evo_graph.nodes), len(graph.nodes), l1_error(evo_graph, graph))
            err[stgy].append(l1_error(evo_graph, graph))

    plt.figure(figsize=(12, 6))
    plt.title("Average L1 error")
    for stgy in ['rd', 'rr', 'wr', 'pr']:
        plt.plot(err[stgy], 'o-', color=color[stgy], label=stgy)
    plt.legend(loc="best")
    # plt.show()
    plt.savefig('analysis.png')

    print('time: ', time.time() - start)


if __name__ == '__main__':
    main()
