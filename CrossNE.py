# encoding: utf8
from utils import readfile, cal_similarity_in_graphs,cal_complementary_set
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from biasDW import load_edgelist, build_deepwalk_corpus, Word2Vec
from score import *


dataset_nodes = {'al':45732, 'aminer':8023, 'SacchPomb':4092, 'SacchCere':6570, 'arXiv': 14489, 'Homo':18222, 'NY':102439, 'Cannes':438537}
parser = ArgumentParser("CrossME",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
parser.add_argument("--alpha", default=0.8, type=float)  # alpha 为 正确的边
parser.add_argument("--beta", default=0.2,type=float)  # beta 为 其他层的信息产生的边，对nosie的忍耐程度

parser.add_argument('--p', default='0.5', type=str)  # p为训练率
parser.add_argument('--dataset', default='arXiv')
parser.add_argument('--korder', default=3, type=int)
parser.add_argument("--dim", default=100,type=int)


args = parser.parse_args()
p = args.p
args.test = 'dataset/'+args.dataset+'/test'+p+'.txt'
args.train = 'dataset/'+args.dataset+'/train'+p+'.txt'
args.layer = 'dataset/'+args.dataset+'/layers.txt'
args.output = args.dataset + '_' + p
# print args.train


alpha = args.alpha
beta = args.beta
dim = args.dim
dataset = args.dataset
num_nodes = dataset_nodes[dataset]

graphs, id2layer = readfile(args.train, args.layer, num_nodes=num_nodes)

adj_dict = {}
layerIDs = graphs.keys()


for id in graphs:
    g = graphs[id]
    A = g.to_sparse_adj_matrix()
    adj_dict[id] = A

E = [ [0 for j in range(len(layerIDs)+1)] for i  in range(len(layerIDs)+1)]
layerid = graphs.keys()
for i in layerid:
    for j in layerid:
        if j>i:
            e_ij, e_ji = cal_complementary_set(adj_dict[i], adj_dict[j])
            E[i][j] = e_ij
            E[j][i] = e_ji


import os
sim_matrix = cal_similarity_in_graphs(adj_dict, num_nodes)



layer_extra_edges = {}
for i in layerid:
    extra_edges_dict = {}
    for j in layerid:
        if j != i:
            node_similarities = sim_matrix[i][j]  # Gi和Gj每个节点的相似度
            rows, cols = E[i][j].nonzero()  # 边在j中且不在i中

            for h in range(len(rows)):
                a = rows[h]
                b = cols[h]
                weight = 1 * node_similarities[a] * node_similarities[b]
                if weight-0.001 < 0:
                    continue
                if (extra_edges_dict.has_key(str(a)+'->'+str(b)) and \
                    extra_edges_dict[str(a)+'->'+str(b)] < weight) or \
                    not extra_edges_dict.has_key(str(a)+'->'+str(b)):
                    extra_edges_dict[str(a)+'->'+str(b)] = float(weight)
    extra_edges = [list(map(int, edges_key.split('->'))) + [extra_edges_dict[edges_key]] for edges_key in extra_edges_dict]
    layer_extra_edges[i] = extra_edges


def _gen_edges(C):
    weights = C.data
    weights = map(float,weights)
    rows, columns = C.nonzero()
    assert len(weights) == len(rows)
    edges = []
    for i in range(len(weights)):
        if rows[i] != columns[i]:
            edges.append([rows[i],columns[i],weights[i]])
    return edges

Embeddings = {}
for layerid in layerIDs:
    import random
    cur_edges = _gen_edges(adj_dict[layerid])
    extra_edges = layer_extra_edges[layerid]

    for i in range(len(cur_edges)):  # 当前网络的边
        cur_edges[i][2] *= alpha

    for i in range(len(extra_edges)):  # 其他网络的边
        extra_edges[i][2] *= beta

    edges = list(cur_edges) + list(extra_edges)


    G = load_edgelist(edges)
    #print 'Walking'
    walks = build_deepwalk_corpus(G, num_paths=20,
                                            path_length=80, alpha=0, rand=random.Random(0))

    #print "Traing"
    model = Word2Vec(walks, size=dim, window=5, min_count=0, sg=1, hs=1, workers=128)

    node2vec = {}
    for key in model.wv.vocab.keys():
        node2vec[int(key)] = model[key]

    Embeddings[layerid] = node2vec


# pickle.dump(Embeddings, open('CrossNE/'+dataset+'_'+str(p)+'_'+str(alpha)+'_'+str(beta)+'.pk', 'wb'))

auc = []
for i in range(1,len(id2layer)+1):
    each_auc =[]
    node2vec = Embeddings
    for _ in range(5):
        each_auc.append(AUC(node2vec[i], 'dataset/'+dataset+'/test'+str(p)+'.txt', i))
    # print(np.mean(each_auc))
    auc.append(np.mean(each_auc))
print(args.dataset, args.alpha, args.beta, np.mean(auc)*100)
