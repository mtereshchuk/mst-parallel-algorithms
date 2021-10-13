import inspect
import re

from enum import Enum
from itertools import chain


class AlgorithmKind(Enum):
    SEQUENTIAL = 0
    PARALLEL = 1


class MstAlgorithmType(Enum):
    PRIM = 0
    KRUSKAL = 1


class Algorithm:
    def name(self):
        class_name = self.__class__.__name__
        return ' '.join(_split_camel_case(class_name))

    def kind(self):
        return self._raise_not_implemented_error()

    def _raise_not_implemented_error(self):
        raise NotImplementedError(f'\'{inspect.stack()[1][3]}\' of \'{self.__class__.__name__}\' is not implemented')


class MstAlgorithm(Algorithm):
    def type(self):
        return self._raise_not_implemented_error()

    def mst(self, g):
        return self._raise_not_implemented_error()


class SequentialAlgorithm(Algorithm):
    def kind(self):
        return AlgorithmKind.SEQUENTIAL


class ParallelAlgorithm(Algorithm):
    def __init__(self, executor):
        self._executor = executor

    def name(self):
        executor_class_name = self._executor.__class__.__name__
        return super().name() + f' ({_split_camel_case(executor_class_name)[0]})'

    def kind(self):
        return AlgorithmKind.PARALLEL

    def _workers_num(self):
        return self._executor._max_workers

    def _parallel_chunks(self, chunk_f, merge_f, arr):
        workers_num = self._workers_num()

        if workers_num < len(arr):
            chunk_size = len(arr) // workers_num
            chunks = [arr[chunk_size * i: chunk_size * (i + 1)] for i in range(workers_num)]
            chunks.append(arr[chunk_size * workers_num:])
        else:
            chunks = [arr]

        chunks = self._executor.map(chunk_f, chunks)
        return self._parallel_merge(merge_f, chunks)

    def _parallel_merge(self, merge_f, chunks):
        for _ in range(self._workers_num()):
            prev = None
            next_data = []
            for chunk in chunks:
                if prev is None:
                    prev = chunk
                else:
                    next_data.append((prev, chunk))
                    prev = None

            chunks = self._executor.map(merge_f, next_data)
            if prev is not None:
                chunks = chain(chunks, [prev])

        return list(chunks)[0]


def merge_chunks(chunks):
    chunk_1, chunk_2 = chunks
    return chunk_1 + chunk_2


def _split_camel_case(s):
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', s)
