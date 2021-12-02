import numpy as np


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []
        self.pagerank = 1.0

    def link_child(self, new_child):
        for child in self.children:
            if child.name == new_child.name:
                return None
        self.children.append(new_child)

    def link_parent(self, new_parent):
        for parent in self.parents:
            if parent.name == new_parent.name:
                return None
        self.parents.append(new_parent)

    def update_pagerank(self, d, n):
        in_neighbors = self.parents
        pagerank_sum = sum((node.pagerank / len(node.children)) for node in in_neighbors)
        random_jumping = d / n
        self.pagerank = random_jumping + (1-d) * pagerank_sum


class Graph:
    def __init__(self, as_name):
        self.as_name = as_name
        # index for round-robin
        self.index = 0
        self.nodes = []
        self.node_names = []

    def contains(self, name):
        return name in self.node_names

    # Return the node with the name, create and return new node if not found
    def find(self, name):
        if not self.contains(name):
            new_node = Node(name)
            self.nodes.append(new_node)
            self.node_names.append(name)
            return new_node
        else:
            return next(node for node in self.nodes if node.name == name)

    def delete_node(self, node):
        assert node in self.nodes

        # TODO
        # recursively delete nodes
        self.nodes.remove(node)
        self.node_names.remove(node.name)
        for child in node.children:
            child.parents.remove(node)
        for parent in node.parents:
            parent.children.remove(node)

    def add_node(self, node):
        # TODO
        # recursively add new nodes
        for child in node.children:

            self.add_edge(node.name, child.name)

            self.add_node(child)
        for parent in node.parents:
            self.add_edge(parent.name, node.name)
            self.add_node(parent)

    def add_edge(self, parent, child):
        if self.as_name == "as-733":
            # Direct the graph
            if parent >= child:
                return None
        parent_node = self.find(parent)
        child_node = self.find(child)
        parent_node.link_child(child_node)
        child_node.link_parent(parent_node)

    def display(self):
        for node in self.nodes:
            print(f'{node.name} links to {[child.name for child in node.children]}')

    def return_top_k(self, k):
        # Sort nodes according to pagerank value
        return sorted(self.nodes, key=lambda node: node.pagerank, reverse=True)[:k]

    def return_rr_k(self, k):
        # Return k nodes round-robin
        self.index += 10
        self.index %= len(self.nodes)
        return self.nodes[self.index - 10: self.index - 10 + k]

    def normalize_pagerank(self):
        pagerank_sum = sum(node.pagerank for node in self.nodes)

        for node in self.nodes:
            node.pagerank /= pagerank_sum

    def clear_pagerank(self):
        for node in self.nodes:
            node.pagerank = 1.0

    def get_pagerank_list(self):
        pagerank_list = np.asarray([node.pagerank for node in self.nodes], dtype='float128')
        return pagerank_list
