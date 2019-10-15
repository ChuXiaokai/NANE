# encoding: utf8
import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict, Iterable, Counter
from scipy import sparse


def f(x):
    return int(x[0]), int(x[1]), float(x[2])


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)
        self.edges = 0

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()]) / 2

    def check_edges(self):
        for x in self.keys():
            self[x] = list(set(self[x]))

    def to_sparse_adj_matrix(self, undirecte=True):
        coo_matrix_sum = sparse.coo_matrix.sum
        edges = np.array(self.edges)
        n1 = edges[:, 0]
        n2 = edges[:, 1]

        w = np.ones(len(n1))
        adj_matrix = sparse.coo_matrix((w, [n1, n2]), shape=[self.num_nodes, self.num_nodes])
        return adj_matrix


def cal_complementary_set(m1, m2):
    """
    m1 and m2 are two binary coo_matrix
    """
    m = m1 + m2
    m[m > 0] = 1
    return m - m1, m - m2


def load_edgelist(edgelist, num_nodes, undirected=True):
    G = Graph()
    G.num_nodes = num_nodes
    edges = []
    for e in edgelist:
        a, b, w = e
        a = int(a)
        b = int(b)
        G[a].append(b)
        edges.append([a, b, w])
        if undirected:
            G[b].append(a)
            edges.append([b, a, w])
    G.edges = edges
    G.check_edges()
    return G


def readfile(graph_path=None, layer_path=None, num_nodes=None):
    layers = pd.read_csv(layer_path, sep=' ').values
    id2layer = dict(layers)

    graphs = pd.read_csv(graph_path, sep=' ', header=None)
    graphs.columns = ["layerID", "n1", "n2", "weight"]
    layers = {}
    for layerID, graph in graphs.groupby(['layerID']):
        edges = graph[['n1', 'n2', 'weight']].values
        edges = map(f, edges)
        layers[layerID] = load_edgelist(edges, num_nodes=num_nodes)
    return layers, id2layer


def cal_k_order(adj_matrix, num_nodes, UNDIRECTED=True):
    D = np.sum(adj_matrix, axis=1).reshape(num_nodes)
    D = np.array(D)[0]
    for i in range(len(D)):
        if (D[i] - 0) > 0.01:
            D[i] = 1 / D[i]

    D = sparse.coo_matrix((D, [range(len(D)), range(len(D))]), shape=[num_nodes, num_nodes])

    # 转移矩阵
    A1 = np.dot(D, adj_matrix)
    A2 = np.dot(A1, A1)
    # K = A1+A2
    A3 = np.dot(A2, A1)

    # 为了获得更好的相似度，我们要把对角线变成0
    data = A1.data
    rows, cols = A1.nonzero()
    assert len(data) == len(rows)
    for i in range(len(rows)):
        if rows[i] == cols[i]:
            data[i] = 0
    A1 = sparse.coo_matrix((data, [rows, cols]), shape=[num_nodes, num_nodes]).tocsr()

    data = A2.data
    rows, cols = A2.nonzero()
    assert len(data) == len(rows)
    for i in range(len(rows)):
        if rows[i] == cols[i]:
            data[i] = 0
    A2 = sparse.coo_matrix((data, [rows, cols]), shape=[num_nodes, num_nodes]).tocsr()

    data = A3.data
    rows, cols = A3.nonzero()
    assert len(data) == len(rows)
    for i in range(len(rows)):
        if rows[i] == cols[i]:
            data[i] = 0
    A3 = sparse.coo_matrix((data, [rows, cols]), shape=[num_nodes, num_nodes]).tocsr()

    return [A1,A2,A3]


def similarity(a, b, normal=False):
    if normal:
        upper = np.sum(np.multiply(a, b), axis=1)
        a_ = np.sum(np.multiply(a, a), axis=1)
        b_ = np.sum(np.multiply(b, b), axis=1)
        down = np.sqrt(a_) * np.sqrt(b_)
        # 去掉0, 就是说该点是孤立点
        down[down == 0] = 1
        return upper / down
    else:
        # 调用系数矩阵对应的操作
        mul = sparse.csr_matrix.multiply
        sum = sparse.csr_matrix.sum

        upper = sum(mul(a, b), axis=1)
        a_ = sum(mul(a, a), axis=1)
        b_ = sum(mul(b, b), axis=1)
        dim_i, dim_j = a_.shape
        a_ = np.array(a_.reshape(dim_i))[0]
        b_ = np.array(b_.reshape(dim_i))[0]
        down = np.sqrt(a_) * np.sqrt(b_)

        # 去掉0
        down[down == 0] = 1
        upper = np.array(upper.reshape(dim_i))[0]
        return upper / down


def cal_similarity_in_graphs(adj_matrixs, num_nodes):
    """
    :param adj_matrixs:  各层网络的adj-matrix
    """

    layerID2KOrder = {}
    for layerID in adj_matrixs:
        adj_m = adj_matrixs[layerID]
        layerID2KOrder[layerID] = cal_k_order(adj_m, num_nodes)

    # 计算每两个矩阵的相似度
    layerIDs = adj_matrixs.keys()
    sim_matrix = [[0 for j in range(len(layerIDs) + 1)] for i in range(len(layerIDs) + 1)]

    for i in layerIDs:
        # Ki = layerID2KOrder[i]
        Ki1, Ki2, Ki3 = layerID2KOrder[i]
        for j in layerIDs:
            if i != j:
                # Kj = layerID2KOrder[j]
                Kj1, Kj2, Kj3 = layerID2KOrder[j]
                sim1 = similarity(Ki1, Kj1)
                sim2 = similarity(Ki2, Kj2)
                sim3 = similarity(Ki3, Kj3)
                sim = 1 * sim1 + 1.0 / 2 * sim2 + 1.0 / 3 * sim3
                sim_matrix[i][j] = sim
    return sim_matrix


def zip_csr_matrix(m):
    """
    m is a sparse matrix
    output is a ziped m
    """
    data = m.data
    rows, columns = m.nonzero()

    count_non_zero = Counter()
    count_non_zero.update(rows)
    count_non_zero.update(columns)
    exist_nodes = count_non_zero.keys()
    newN2oldN = dict([(i, n) for i, n in enumerate(exist_nodes)])
    oldN2newN = dict([(n, i) for i, n in enumerate(exist_nodes)])
    new_rows = [oldN2newN[index] for index in rows]
    new_columns = [oldN2newN[index] for index in columns]
    return sparse.csr_matrix((data, (new_rows, new_columns)),
                             shape=(len(exist_nodes), len(exist_nodes))), newN2oldN, sorted(oldN2newN.keys()), len(
        newN2oldN.keys())
