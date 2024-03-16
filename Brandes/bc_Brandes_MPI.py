import struct
import sys
import time
from struct import unpack as un

import numpy as np
from mpi4py import MPI

"""
python Graph.py csr res
"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Запуск нового таймера"""
        self._start_time = time.perf_counter()

    def stop(self):
        """Отстановить таймер и сообщить о времени вычисления"""
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Вычисление заняло {elapsed_time:0.4f} секунд")


class MyGraph:
    number_of_nodes = None  # количество вершин
    number_of_edges = None  # количество граней
    rowsIndices = []  # индексы начала граней
    endV = []  # индексы концов граней

    def __init__(self, source):
        self.source = source

        with open(self.source, "rb") as f:
            graph_bin_format = f.read()
        self.number_of_nodes, = un("I", graph_bin_format[:4])
        self.number_of_edges, = un("Q", graph_bin_format[4:12])
        buffer, = un("B", graph_bin_format[12:13])
        for i in range(0, (self.number_of_nodes + 1) * 8, 8):
            self.rowsIndices += un("Q", graph_bin_format[13 + i: 21 + i])
        for i in range((self.number_of_nodes + 1) * 8 + 13, len(graph_bin_format), 4):
            self.endV += un("I", graph_bin_format[i: i + 4])

    def find_neighbors(self, node):
        # Найдем индекс начала и конца строки, соответствующей вершине node
        start = self.rowsIndices[node]
        end = self.rowsIndices[node + 1] \
            if node < len(self.rowsIndices) - 1 \
            else len(self.endV)
        # Вернем все вершины, которые являются соседями вершины node
        return self.endV[start:end]

    def processing_single_node(self, node):
        # BrandesInitS
        BC = np.zeros(self.number_of_nodes)
        P = [[] for x in range(self.number_of_nodes)]
        Stack = []
        Queue = [node]
        sigma = np.zeros(self.number_of_nodes)
        delta = np.zeros(self.number_of_nodes)
        d = np.full(self.number_of_nodes, -1)
        sigma[node] = 1
        d[node] = 0

        # ShortestPathCounting
        while Queue:
            v = Queue.pop(0)
            Stack.append(v)
            for w in self.find_neighbors(v):
                if d[w] < 0:
                    Queue.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # DependencyAccumulation
        while Stack:
            w = Stack.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != node:
                BC[w] += 0.5 * delta[w]
        return BC


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    timer = Timer()
    timer.start()

csr = sys.argv[1]
res = sys.argv[2]
graph = MyGraph(csr)
nodes_for_one_process = graph.number_of_nodes // size

if rank == size - 1:
    data = np.sum([graph.processing_single_node(i) for i in range(rank * nodes_for_one_process,
                                                                  graph.number_of_nodes)], axis=0)
else:
    data = np.sum([graph.processing_single_node(i) for i in range(rank * nodes_for_one_process, (rank + 1) *
                                                                  nodes_for_one_process)], axis=0)

recvdata = np.zeros(graph.number_of_nodes)
comm.Reduce(data, recvdata, root=0, op=MPI.SUM)

if rank == 0:
    result = recvdata
    timer.stop()
    with open(res, "wb") as f:
        for elem in result:
            f.write(struct.pack("d", elem))
