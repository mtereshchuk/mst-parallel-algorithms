import os

import networkx as nx

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from algorithm import MstAlgorithm, ParallelAlgorithm, MstAlgorithmType, SequentialAlgorithm
from graph import gen_random_graph, true_mst, edges_sum


class BasePrim(MstAlgorithm):
    def type(self):
        return MstAlgorithmType.PRIM

    def mst(self, g):
        r = 0

        nodes = g.nodes()
        used = [False for _ in nodes]
        d = [float('inf') for _ in nodes]
        p = [-1 for _ in nodes]

        d[r] = 0
        for _ in nodes:
            u, _ = self._find_min(used, d)
            used[u] = True

            for _, v, data in g.edges(u, data=True):
                w = data['weight']
                if not used[v] and w < d[v]:
                    d[v] = w
                    p[v] = u

        return [(v, p[v], g.get_edge_data(v, p[v])) for v in nodes if p[v] != -1]

    def _find_min(self, used, d):
        return self._raise_not_implemented_error()


class SequentialPrim(BasePrim, SequentialAlgorithm):
    def _find_min(self, used, d):
        u = -1
        for v in range(len(used)):
            if not used[v] and (u == -1 or d[v] < d[u]):
                u = v
        return u, d[u]


class ParallelPrim(BasePrim, ParallelAlgorithm):
    def __init__(self, executor):
        super().__init__(executor)

    def _find_min(self, used, d):
        return self._parallel_chunks(partial(_mst_chunk_f, used), _mst_merge_f, list(enumerate(d)))


def _mst_chunk_f(used, chunk):
    not_used = list(filter(lambda p: not used[p[0]], chunk))
    if len(not_used) == 0:
        return -1, float('inf')

    return min(not_used, key=lambda p: p[1])


def _mst_merge_f(chunks):
    chunk_1, chunk_2 = chunks
    return min(chunk_1, chunk_2, key=lambda p: p[1])


if __name__ == '__main__':
    test_g = gen_random_graph(100, 500)
    a = nx.adjacency_matrix(test_g)

    print(edges_sum(true_mst(test_g)))

    sequentialPrim = SequentialPrim()
    print(edges_sum(sequentialPrim.mst(test_g)))

    with ThreadPoolExecutor(max_workers=os.cpu_count() + 4) as executor:
        parallelPrim = ParallelPrim(executor)
        print(edges_sum(parallelPrim.mst(test_g)))

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        parallelPrim = ParallelPrim(executor)
        print(edges_sum(parallelPrim.mst(test_g)))
