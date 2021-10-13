import sys

import os
import time
import toml
import logging
import numpy as np

from matplotlib import pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from graph import DensityType, gen_random_connected_graph
from kruskal import SequentialKruskal, SequentialFilterKruskal, ParallelKruskal, ParallelFilterKruskal
from prim import SequentialPrim, ParallelPrim

CONFIG_FILE = 'config.toml'


def main():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    config = toml.load(CONFIG_FILE)
    cpu_count = os.cpu_count()

    with ThreadPoolExecutor(max_workers=cpu_count + config['threads']['number_add']) as thread_pool, \
            ProcessPoolExecutor(max_workers=cpu_count) as process_pool:

        vertices_cfg = config['vertices']
        vertices_number_range = range(0, vertices_cfg['max_number'] + 1, vertices_cfg['number_step'])

        kruskal_filter_threshold_part = config['kruskal']['filter']['threshold_part']
        algorithms = [
            SequentialPrim(),
            ParallelPrim(thread_pool),
            ParallelPrim(process_pool),

            SequentialKruskal(),
            ParallelKruskal(thread_pool),
            ParallelKruskal(process_pool),

            SequentialFilterKruskal(kruskal_filter_threshold_part),
            ParallelFilterKruskal(thread_pool, kruskal_filter_threshold_part),
            ParallelFilterKruskal(process_pool, kruskal_filter_threshold_part)
        ]

        logging.info('Analysis has begun')
        for densityType in DensityType:
            edges_cfg = config['edges']

            density_type_name = densityType.name.lower()
            logging.info(f'Started for {density_type_name} graphs')

            density_cfg = edges_cfg[f'{density_type_name}']
            min_offset, max_offset = density_cfg['min_offset'], density_cfg['max_offset']

            results = defaultdict(lambda: [])
            for n_ in vertices_number_range:
                if n_ == 0:
                    n = vertices_cfg['min_number']
                else:
                    n = n_

                m = densityType.random_edges_num(min_offset, max_offset, n)
                logging.debug(f'Test graph with {n} vertices and {m} edges')

                g = gen_random_connected_graph(n, m, edges_cfg['rewiring_prob'], edges_cfg['min_weight'],
                                               edges_cfg['max_weight'])

                for algorithm in algorithms:
                    logging.debug(f'Test {algorithm.name()}')

                    attempt_results = []
                    for _ in range(config['execution']['repeat']):
                        start = time.time()
                        algorithm.mst(g)
                        end = time.time()
                        attempt_results.append(end - start)

                    mean_time = np.mean(attempt_results)
                    results[algorithm.name()].append(mean_time)

            plt.figure(figsize=(config['plot']['length'], config['plot']['width']))

            vertices_numbers = list(vertices_number_range)
            vertices_numbers[0] = vertices_cfg['min_number']
            for name, times in results.items():
                plt.plot(vertices_number_range, times, label=name)

            plt.title(f'{density_type_name.capitalize()} graphs')
            plt.xlabel('Number of vertices')
            plt.ylabel('Time in seconds')
            plt.legend()
            plt.savefig(f'{density_type_name}-graphs.png')


if __name__ == '__main__':
    main()
