# import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import *

# sys.setrecursionlimit(10000)

# https://github.com/chonyy/PageRank-HITS-SimRank

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='as-733', help='dataset(as-733/as-caida)')
parser.add_argument('--damping_factor', type=float, default=0.15, help='damping factor')
parser.add_argument('--iterations', type=int, default=500, help='iterations')
parser.add_argument('--sequence_length', type=int, default=100, help='graph sequence length')
parser.add_argument('--probing_nodes_num', type=int, default=10, help='number of probing nodes')
parser.add_argument('--fig_path', type=str, default='analysis.png', help='log fig save path')
opt = parser.parse_args()


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

    PageRank(g, opt.damping_factor, opt.iterations)
    # g.display()
    print(g.get_pagerank_list())
    # [2          1          3]
    # [0.28674488 0.10375129 0.60950383]

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

    g.display()
    print()
    g2.display()

    PageRank(g, opt.damping_factor, opt.iterations)
    print(g.get_pagerank_list())
    PageRank(g2, opt.damping_factor, opt.iterations)
    print(g2.get_pagerank_list())
    # [5          2          1          4          3]
    # [0.30867546 0.18550762 0.12568849 0.14755548 0.23257295]

    print(l1_error(g, g2))


def probing(logger, as_lines):
    graphs = []
    for i in tqdm(range(1, len(as_lines))):
        graph = build_graph(as_lines[i], opt.dataset)
        # print(len(graph.nodes))
        PageRank(graph, opt.damping_factor, opt.iterations)
        graphs.append(graph)
    logger.info('Building graphs done')

    edge_nums = [graph.edges for graph in graphs]
    # node_nums = [len(graph.nodes) for graph in graphs]
    logger.info('Edges gradient average with nodes fixed: {:.3f}'.format(np.mean(list(map(abs, np.gradient(edge_nums))))))

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(len(graphs)), edge_nums, '*-', color='r', label="edges")
    # plt.plot(np.arange(len(graphs)), node_nums, 'o-', color='b', label="nodes")
    plt.locator_params(axis="y", nbins=10)
    plt.legend(loc="best")
    plt.savefig('edge_statistics.png')

    err = defaultdict(list)
    color = {'rd': 'b', 'rr': 'r', 'wr': 'g', 'pr': 'y'}

    for stgy in ['rd', 'rr', 'wr', 'pr']:
        evo_graph = build_graph(as_lines[0], opt.dataset)
        # print(len(evo_graph.nodes))
        PageRank(evo_graph, opt.damping_factor, opt.iterations)
        # evo_graph.init_priority()

        logger.info('Sum of page rank: {}'.format(sum(evo_graph.get_pagerank_list())))

        evo_bar = tqdm(range(len(graphs)))
        for i in evo_bar:
            # print(len(evo_graph.nodes))
            evolve_graph(evo_graph, graphs[i], stgy, opt.probing_nodes_num, opt.damping_factor, opt.iterations)
            error = l1_error(evo_graph, graphs[i])
            evo_bar.set_description(
                'Stratergy: [{}] L1 error: {:e}'.format(stgy, error))
            err[stgy].append(error)

    plt.figure(figsize=(12, 6))
    plt.title("Average L1 error")
    for stgy in ['rd', 'rr', 'wr', 'pr']:
        plt.plot(err[stgy], 'o-', color=color[stgy], label=stgy)
    plt.legend(loc="best")
    plt.savefig(opt.fig_path)


def main():
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if opt.dataset == 'as-733':
        # undirected with loop
        dsroot = "../data/as-733"
        skip_lines = 2
        thresh = 100
    elif opt.dataset == 'as-caida':
        # directed without loop
        dsroot = "../data/as-caida"
        skip_lines = 6
        thresh = 500
    else:
        logger.info('Invaild dataset')
        return None

    graphnames = [f for f in os.listdir(dsroot) if os.path.isfile(os.path.join(dsroot, f))]

    logger.info('Parsing {}'.format(opt.dataset))
    as_nodes, as_edges, as_lines = parse_as(dsroot, graphnames, skip_lines=skip_lines, num=opt.sequence_length, thresh=thresh)
    logger.info('Nodes gradient average: {:.3f}'.format(np.mean(list(map(abs, np.gradient(as_nodes))))))
    logger.info('Edges gradient average: {:.3f}'.format(np.mean(list(map(abs, np.gradient(as_edges))))))
    logger.info('Dataset sequence length : {}'.format(len(as_lines)))
    logger.info('Parsing {} done'.format(opt.dataset))

    o_nodes = get_nodes(as_lines[0])
    for lines in as_lines[1:]:
        o_nodes &= get_nodes(lines)

    as_evo_lines = parse_nodes_constant(as_lines, o_nodes)

    logger.info('Static nodes number: {}'.format(len(o_nodes)))

    # for i in tqdm(range(len(as_evo_lines))):
    #     graph = build_graph(as_evo_lines[i], opt.dataset)
    #     o_nodes &= set(graph.node_names)

    probing(logger, as_evo_lines)


if __name__ == '__main__':
    # test()
    main()
