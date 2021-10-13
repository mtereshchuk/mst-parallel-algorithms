import random
import networkx as nx

from matplotlib import pyplot as plt
from merge_sort import merge_sort
from enum import Enum


class DensityType(Enum):
    # SPARSE = 0
    # MEDIUM = 1
    DENSE = 2

    def random_edges_num(self, min_offset, max_offset, n):
        offset = random.uniform(min_offset, max_offset)

        # if self == self.SPARSE:
        #     return int(self._min_edge_num(n) * (1 + offset))

        # if self == self.MEDIUM:
        #     medium_offset = 1 - offset if bool(random.getrandbits(1)) else 1 + offset
        #     return int(((self._max_edge_num(n) - self._min_edge_num(n)) / 2) * medium_offset)

        return int(self._max_edge_num(n) * (1 - offset))

    def _min_edge_num(self, n):
        return n - 1

    def _max_edge_num(self, n):
        return (n * (n - 1)) // 2


def gen_random_graph(n, m, w_from=1, w_to=10000000):
    g = nx.gnm_random_graph(n=n, m=m)
    return with_random_weights(g, w_from, w_to)


def gen_random_connected_graph(n, m, p, w_from, w_to):
    g = nx.connected_watts_strogatz_graph(n=n, k=int(m / n) + 1, p=p)
    return with_random_weights(g, w_from, w_to)


def with_random_weights(g, w_from, w_to):
    weights = random.sample(range(w_from, w_to + 1), g.number_of_edges())
    for i, e in enumerate(g.edges(data=True)):
        e[2]['weight'] = weights[i]
    return g


def plot_graph(g):
    pos = nx.nx_agraph.graphviz_layout(g)
    nx.draw(g, pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.show()


def edges_from_graph(g):
    return list(g.edges(data=True))


def edges_sum(edges):
    return sum(e[2]['weight'] for e in edges)


def true_mst(g):
    return list(nx.minimum_spanning_tree(g).edges(data=True))


def sorted_edges(edges):
    return merge_sort(edges, key=lambda e: e[2]['weight'])
