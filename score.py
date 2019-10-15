# encoding:utf8
import pickle
import numpy as np
import random
def AUC(node2vec, test_graph, i):
    """
    :param node2vec:  type = dict
    :param test_graph:
    """

    # print np.dot(node2vec[1], node2vec[0])
    # print np.dot(node2vec[1], node2vec[2])
    # print np.dot(node2vec[1], node2vec[3])

    f1 = open(test_graph, 'rb').readlines()
    edges = []
    for line in f1:
        index, a,b,w = line.split()
        if index == str(i):
            edges.append(map(int,[a,b]))

    nodes = list(set([i for j in edges for i in j]))
    a = 0
    b = 0
    for i, j in edges:
        if node2vec.has_key(i) and node2vec.has_key(j):
            dot1 = np.dot(node2vec[i], node2vec[j])
            random_node = random.sample(nodes, 1)[0]
            while random_node == j or (not node2vec.has_key(random_node)):
                random_node = random.sample(nodes, 1)[0]
            dot2 = np.dot(node2vec[i], node2vec[random_node])
            if dot1 > dot2:
                a += 1
            elif dot1 == dot2:
                a += 0.5
            b += 1

    return float(a) / b
