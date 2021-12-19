# import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import *
from unit_test import pg_test, evolve_test

# https://github.com/chonyy/PageRank-HITS-SimRank

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='as-733', help='dataset(as-733/as-caida)')
parser.add_argument('--damping_factor', type=float, default=0.15, help='damping factor')
parser.add_argument('--iterations', type=int, default=500, help='iterations')
parser.add_argument('--sequence_length', type=int, default=100, help='graph sequence length')
parser.add_argument('--probing_nodes_num', type=int, default=10, help='number of probing nodes')
parser.add_argument('--fig_path', type=str, default='analysis.png', help='log fig save path')
opt = parser.parse_args()


def probing_main(logger, as_edges):
    # Build graph sequence
    graphs = []
    for i in tqdm(range(1, len(as_edges))):
        graph = build_graph(as_edges[i], opt.dataset)
        # print(len(graph.nodes))
        PageRank(graph, opt.damping_factor, opt.iterations)
        graphs.append(graph)
    logger.info('Building graphs done')

    err = defaultdict(list)
    color = {'rd': 'b', 'rr': 'r', 'wr': 'g', 'pr': 'y'}

    '''
        Evaluating four algorithms
        rd = Random Probing
        rr = Round-Robin Probing
        wr = Proportional Probing
        pr = Priority Probing
    '''
    for stgy in ['rd', 'rr', 'wr', 'pr']:
        # Build initial graph
        evo_graph = build_graph(as_edges[0], opt.dataset)
        # print(len(evo_graph.nodes))
        PageRank(evo_graph, opt.damping_factor, opt.iterations)

        logger.info('Sum of page rank: {}'.format(sum(evo_graph.get_pagerank_list())))

        evo_bar = tqdm(range(len(graphs)))
        for i in evo_bar:
            evolve_graph(evo_graph, graphs[i],
                         stgy, opt.probing_nodes_num, opt.damping_factor, opt.iterations)
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
        # Undirected with loop
        dsroot = "../data/as-733"
        skip_lines, thresh = 2, 100
    elif opt.dataset == 'as-caida':
        # Directed without loop
        dsroot = "../data/as-caida"
        skip_lines, thresh = 6, 500
    else:
        logger.info('Invaild dataset')
        return None

    graphnames = [f for f in os.listdir(dsroot) if os.path.isfile(os.path.join(dsroot, f))]

    logger.info('Parsing {}'.format(opt.dataset))
    as_node_nums, as_edge_nums, as_edges = parse_as(dsroot, graphnames, skip_lines=skip_lines, num=opt.sequence_length,
                                                    thresh=thresh)
    logger.info('Nodes gradient average: {:.3f}'.format(np.mean(list(map(abs, np.gradient(as_node_nums))))))
    logger.info('Edges gradient average: {:.3f}'.format(np.mean(list(map(abs, np.gradient(as_edge_nums))))))
    logger.info('Dataset sequence length: {}'.format(len(as_edges)))
    logger.info('Parsing {} done'.format(opt.dataset))

    '''Nodes do not change'''
    # Get common nodes
    common_nodes = get_nodes(as_edges[0])
    for lines in as_edges[1:]:
        common_nodes &= get_nodes(lines)
    # Fix nodes number
    as_fix_edges = parse_nodes_constant(as_edges, common_nodes)
    logger.info('Fix nodes number: {}'.format(len(common_nodes)))
    probing_main(logger, as_fix_edges)

    # plt.figure(figsize=(12, 6))
    # plt.plot(np.arange(len(as_edge_nums)), as_edge_nums, '*-', color='r', label="edges")
    # plt.plot(np.arange(len(as_node_nums)), as_node_nums, 'o-', color='b', label="nodes")
    # plt.locator_params(axis="y", nbins=10)
    # plt.legend(loc="best")
    # plt.savefig('caida_stat.png')

    '''Nodes change'''
    # probing_main(logger, as_edges)


if __name__ == '__main__':
    # pg_test()
    # evolve_test()
    main()
