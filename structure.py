import numpy as np


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []
        self.pagerank = 1.0
        self.priority = 0.0
        self.new_pagerank = self.pagerank

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
        self.new_pagerank = random_jumping + (1 - d) * pagerank_sum


class Graph:
    def __init__(self, as_name=""):
        self.as_name = as_name
        # Index for round-robin
        self.index = 0
        self.edges = 0
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

        self.nodes.remove(node)
        self.node_names.remove(node.name)
        for child in node.children:
            child.parents.remove(node)
            self.edges -= 1
        for parent in node.parents:
            parent.children.remove(node)
            self.edges -= 1

    def add_node(self, probing_node, true_graph, depth=10):
        true_node = true_graph.find(probing_node.name)

        for child in true_node.children:
            if child.name not in self.node_names and depth > 0:
                self.add_node(child, true_graph, depth - 1)
            if self.find(child.name) not in probing_node.children:
                self.add_edge(probing_node.name, child.name)

        for child in probing_node.children:
            if child.name not in true_graph.node_names:
                self.delete_node(child)
            elif true_graph.find(child.name) not in true_node.children:
                self.delete_edge(probing_node.name, child.name)

        # # Parent probing
        # for parent in true_node.parents:
        #     if parent.name not in self.node_names and depth > 0:
        #         self.add_node(parent, true_graph, depth - 1)
        #     self.add_edge(parent.name, probing_node.name)

    def delete_edge(self, parent, child):
        parent_node = self.find(parent)
        child_node = self.find(child)
        parent_node.children.remove(child_node)
        child_node.parents.remove(parent_node)
        self.edges -= 1

    def add_edge(self, parent, child):
        if self.as_name == "as-733":
            # Direct the graph
            if parent >= child:
                return None
        parent_node = self.find(parent)
        child_node = self.find(child)
        parent_node.link_child(child_node)
        child_node.link_parent(parent_node)
        self.edges += 1

    def display(self):
        for node in self.nodes:
            print(f'{node.name} links to {[child.name for child in node.children]}')

    def return_top_k(self, k):
        # Sort nodes according to priority
        return sorted(self.nodes, key=lambda node: node.priority, reverse=True)[:k]

    def return_rr_k(self, k):
        # Return k nodes round-robin
        if k >= len(self.nodes):
            return self.nodes
        cur_index = self.index
        self.index += k
        self.index %= len(self.nodes)
        if cur_index + k < len(self.nodes):
            return self.nodes[cur_index: cur_index + k]
        return self.nodes[:cur_index + k - len(self.nodes)] + self.nodes[cur_index:]

    def normalize_pagerank(self):
        pagerank_sum = sum(node.pagerank for node in self.nodes)
        for node in self.nodes:
            node.pagerank /= pagerank_sum

    def update_pagerank(self):
        for node in self.nodes:
            node.pagerank = node.new_pagerank

    def clear_pagerank(self):
        for node in self.nodes:
            node.pagerank = 1.0

    def get_pagerank_list(self):
        pagerank_list = np.asarray([node.pagerank for node in self.nodes], dtype='float128')
        return pagerank_list

    def prune(self):
        for node in self.nodes:
            if not node.parents and not node.children:
                self.nodes.remove(node)
                self.node_names.remove(node.name)
