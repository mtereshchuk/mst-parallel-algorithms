import random

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from disjoint_set import DisjointSet
from algorithm import MstAlgorithm, ParallelAlgorithm, merge_chunks, SequentialAlgorithm, MstAlgorithmType
from graph import gen_random_graph, edges_sum, true_mst, edges_from_graph, sorted_edges
from merge_sort import merge, merge_sort


class BaseKruskal(MstAlgorithm):
    def type(self):
        return MstAlgorithmType.KRUSKAL

    def mst(self, g):
        return self._mst_from_edges(edges_from_graph(g))

    def _mst_from_edges(self, edges):
        return self._base_mst_from_edges(edges, DisjointSet())

    def _base_mst_from_edges(self, edges, ds):
        edges = self._sort_edges(edges)
        mst = []
        for u, v, w in edges:
            if not ds.connected(u, v):
                mst.append((u, v, w))
                ds.union(u, v)
        return mst

    def _sort_edges(self, edges):
        return self._handle_chunks(_sort_edges_chunk_f, _sort_edges_merge_f, edges)

    def _handle_chunks(self, chunk_f, merge_f, arr):
        # print('BaseKruskal._handle_chunks')
        return self._raise_not_implemented_error()


class SequentialKruskal(BaseKruskal, SequentialAlgorithm):
    def _handle_chunks(self, chunk_f, merge_f, arr):
        # print('SequentialKruskal._handle_chunks')
        return chunk_f(arr)


class ParallelKruskal(BaseKruskal, ParallelAlgorithm):
    def __init__(self, executor):
        super().__init__(executor)

    def _handle_chunks(self, chunk_f, merge_f, arr):
        # print('ParallelKruskal._handle_chunks')
        return self._parallel_chunks(chunk_f, merge_f, arr)


class FilterKruskal(BaseKruskal):
    _MIN_THRESHOLD_PART = 10

    def __init__(self, threshold_part=0.005):
        self._threshold_part = threshold_part

    def _mst_from_edges(self, edges):
        ds = DisjointSet()
        threshold = max(self._MIN_THRESHOLD_PART, int(len(edges) * self._threshold_part))
        return self._filter_mst_from_edges(edges, ds, threshold)

    def _filter_mst_from_edges(self, edges, ds, threshold):
        edges_num = len(edges)
        if edges_num == 0:
            return []

        if edges_num <= threshold:
            return super()._base_mst_from_edges(edges, ds)

        pivot = random.choice(edges)[2]['weight']
        le, gr = self._partition(edges, pivot)

        le = self._filter_mst_from_edges(le, ds, threshold)
        gr = self._filter(gr, ds)

        return le + self._filter_mst_from_edges(gr, ds, threshold)

    def _partition(self, edges, pivot):
        return self._handle_chunks(partial(_partition_chunk_f, pivot), _partition_merge_f, edges)

    def _filter(self, edges, ds):
        return self._handle_chunks(partial(_filter_chunk_f, ds), _filter_merge_f, edges)


def _partition_chunk_f(pivot, chunk):
    le, gr = [], []
    for u, v, data in chunk:
        e = (u, v, data)
        w = data['weight']
        if w <= pivot:
            le.append(e)
        else:
            gr.append(e)
    return le, gr


def _partition_merge_f(chunks):
    (le_1, gr_1), (le_2, gr_2) = chunks
    return le_1 + le_2, gr_1 + gr_2


def _filter_chunk_f(ds, chunk):
    filtered = []
    for u, v, w in chunk:
        if not ds.connected(u, v):
            filtered.append((u, v, w))
    return filtered


def _filter_merge_f(chunks):
    return merge_chunks(chunks)


def _sort_edges_chunk_f(chunk):
    return merge_sort(chunk, key=lambda e: e[2]['weight'])


def _sort_edges_merge_f(chunks):
    return merge(chunks, key=lambda e: e[2]['weight'])


class SequentialFilterKruskal(FilterKruskal, SequentialKruskal):
    pass


class ParallelFilterKruskal(FilterKruskal, ParallelKruskal):
    def __init__(self, executor, threshold_part=0.005):
        super().__init__(threshold_part)
        self._executor = executor


if __name__ == '__main__':
    test_g = gen_random_graph(100, 100 * 100 // 2)

    print(edges_sum(true_mst(test_g)))

    print('SequentialKruskal')
    sequentialKruskal = SequentialKruskal()
    print(edges_sum(sequentialKruskal.mst(test_g)))

    with ThreadPoolExecutor(max_workers=15) as thread_executor:
        print('ParallelKruskal')
        parallelKruskal = ParallelKruskal(thread_executor)
        print(edges_sum(parallelKruskal.mst(test_g)))

    with ProcessPoolExecutor(max_workers=3) as process_executor:
        print('ParallelKruskal')
        parallelKruskal = ParallelKruskal(process_executor)
        print(edges_sum(parallelKruskal.mst(test_g)))

    print('SequentialFilterKruskal')
    sequentialFilterKruskal = SequentialFilterKruskal()
    print(edges_sum(sequentialFilterKruskal.mst(test_g)))

    with ThreadPoolExecutor(max_workers=15) as thread_executor:
        print('ParallelFilterKruskal')
        parallelFilterKruskal = ParallelFilterKruskal(thread_executor)
        print(edges_sum(parallelFilterKruskal.mst(test_g)))

    with ProcessPoolExecutor(max_workers=3) as process_executor:
        print('ParallelFilterKruskal')
        parallelFilterKruskal = ParallelFilterKruskal(process_executor)
        print(edges_sum(parallelFilterKruskal.mst(test_g)))
