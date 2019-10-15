# encoding: utf8
# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refer to https://github.com/phanein/deepwalks
"""

import logging
from time import time
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
import numpy as np
from gensim.models import Word2Vec

from multiprocessing import cpu_count

logger = logging.getLogger("deepwalk")


class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):

        t0 = time()

        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        logger.info('make_directed: added missing edges {}s'.format(t1 - t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1 - t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()

        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

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


    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.

            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    alias_n = G[cur][alias_draw(J=G.cur_weights[cur][0],q=G.cur_weights[cur][1])]
                    path.append(alias_n)
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


    def make_jump_prob(self, edges):
        """
        根据权重矩阵，设计跳转概率
        jump_prob = {
            n1:{n2:8.1,n3:0.1}
        }
        """
        jump_prob = dict([[key, dict()] for key in self])
        rows, columns, weights = zip(*edges)
        for i in range(len(rows)):
            jump_prob[rows[i]][columns[i]] = weights[i]
        
        for key in jump_prob:
            neighbors_w = jump_prob[key].values()
            neighbors = jump_prob[key].keys()  # 该节点的邻居

            base = np.sum(neighbors_w)
            for n in neighbors:
                jump_prob[key][n] /= base

        self.cur_weights = dict([[key, 0] for key in self])
        for key in self:
            weights = []
            neighbors = self[key]
            for n in neighbors:
                weights.append(jump_prob[key][n])

            J, q = alias_setup(weights)
            self.cur_weights[key] = [J,q]
        


# TODO add build_walks in here
def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
    return walks


def selected_deepwalk_corpus(G, num_paths, path_length, alpha=0, started_nodes=[],
                          rand=random.Random(0)):
    """
    only select some nodes to started
    """
    walks = []

    nodes = started_nodes
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        import time
        for node in nodes:
            walks.append(G.random_walk_old(path_length, rand=rand, alpha=alpha, start=node))
        # print b-a
    return walks


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def load_edgelist(edges, undirected=True, if_jump=True):
    # 默认edges 是无向图
    # W是权重矩阵，默认是sparse.csr格式的
    G = Graph()
    for e in edges:
        x, y, w = e
        G[x].append(y)
    G.make_consistent()
    if if_jump:
        G.make_jump_prob(edges)
    return G


def alias_setup(probs):
    '''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

class Skipgram(Word2Vec):
    """A subclass to allow more customization of the Word2Vec internals."""

    def __init__(self, vocabulary_counts=None, **kwargs):

        self.vocabulary_counts = None

        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["workers"] = kwargs.get("workers", cpu_count())
        kwargs["size"] = kwargs.get("size", 128)
        kwargs["sentences"] = kwargs.get("sentences", None)
        kwargs["window"] = kwargs.get("window", 10)
        kwargs["sg"] = 1
        kwargs["hs"] = 1

        if vocabulary_counts != None:
          self.vocabulary_counts = vocabulary_counts

        super(Skipgram, self).__init__(**kwargs)