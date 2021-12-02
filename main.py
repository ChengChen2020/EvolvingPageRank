import time

from utils import *

# https://github.com/chonyy/PageRank-HITS-SimRank

# undirected
# with loop
dsroot_1 = "/Users/julius/Desktop/courses/NYU/CS6913-WSE/PageRank/as-733/"
# directed
# no loop
dsroot_2 = "/Users/julius/Desktop/courses/NYU/CS6913-WSE/PageRank/as-caida/"

graphnames_1 = [f for f in os.listdir(dsroot_1) if os.path.isfile(os.path.join(dsroot_1, f))]
graphnames_2 = [f for f in os.listdir(dsroot_2) if os.path.isfile(os.path.join(dsroot_2, f))]

assert (len(graphnames_1) == 733)
assert (len(graphnames_2) == 122)

damping_factor = 0.15
iterations = 500
probing_nodes_num = 10  # propotional to current number of nodes

def main():
    # Parse CAIDA
    # caida_nodes, caida_edges, caida_lines = parse_as(dsroot_2, graphnames_2, skip_lines=6)
    # Parse as-733
    print('Parsing as-733')
    as_nodes, as_edges, as_lines = parse_as(dsroot_1, graphnames_1, skip_lines=2)
    print('Parsing as-733 done')

    start = time.time()
    graphs = [build_graph(as_lines[i], 'as-733') for i in range(1, 10)]
    print('time: ', time.time() - start)

    for graph in graphs:
        PageRank(graph, damping_factor, iterations)

    for stgy in ['rd', 'rr', 'wr', 'pr']:

        evo_graph = build_graph(as_lines[0], 'as-733')

        for graph in graphs:
            evolve_graph(evo_graph, graph, stgy)
            PageRank(evo_graph, damping_factor, iterations)
            print(stgy, len(evo_graph.nodes), l1_error(evo_graph, graph))


if __name__ == '__main__':
    main()
