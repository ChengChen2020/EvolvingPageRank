from utils import *
from structure import Graph


def pg_test():
    t = Graph()
    t.add_edge('D', 'A')
    t.add_edge('D', 'B')
    t.add_edge('C', 'B')
    t.add_edge('B', 'C')
    t.add_edge('F', 'E')
    t.add_edge('E', 'F')
    t.add_edge('E', 'D')
    t.add_edge('E', 'B')
    t.add_edge('F', 'B')
    t.add_edge('G', 'B')
    t.add_edge('G', 'E')
    t.add_edge('H', 'B')
    t.add_edge('H', 'E')
    t.add_edge('I', 'B')
    t.add_edge('I', 'E')
    t.add_edge('J', 'E')
    t.add_edge('K', 'E')
    t.display()
    PageRank(t, 0.15, 500)
    print(t.get_pagerank_list())
    print([n.name for n in t.find('B').parents])
    print([n.name for n in t.find('B').children])

    print()

    h = Graph()
    h.add_edge(1, 3)
    h.add_edge(3, 1)
    h.add_edge(3, 2)
    h.add_edge(2, 3)
    h.add_edge(3, 4)
    h.add_edge(1, 4)
    h.add_edge(4, 2)
    h.display()
    PageRank(h, 0.15, 500)
    print(h.get_pagerank_list())


def evolve_test():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    random.shuffle(g.nodes)

    PageRank(g, 0.15, 500)
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

    evolve_graph(g2, g, 'rr', 2, 0.15, 500)

    g.display()
    print()
    g2.display()

    PageRank(g, 0.15, 500)
    print(g.get_pagerank_list())
    PageRank(g2, 0.15, 500)
    print(g2.get_pagerank_list())
    # [5          2          1          4          3]
    # [0.30867546 0.18550762 0.12568849 0.14755548 0.23257295]

    print(l1_error(g, g2))
