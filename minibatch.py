#!/usr/bin/env python3

import numpy as np
import collections
from functools import reduce

def build_batch_from_edges(num_layers, edges, neigh_dict, sample_size, neg_size):
    """
    This batch method is used for unsupervised mode. First, it prepares
    auxiliary matrices for the combination of neighbor nodes (read from edges)
    and negative sample nodes. Second, it provides mappings to filter the
    results into three portions for use in the unsupervised loss function.

    :param int num_layers: number of layers
    :param [(int, int)] edges: edge with node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param int sample_size: sample size
    :param int neg_size: size of batchN
    :return namedtuple minibatch (3 more additional elements to supervised mode)
        "src_nodes": node ids to retrieve from raw feature and feed to the first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
        "dst2batchA": select batchA nodes from all nodes at the last layer
        "dst2batchB": select batchB nodes from all nodes at the last layer
        "dst2batchN": filter batchN nodes from all nodes at the last layer

    Terms:
    - batchA: just a set of nodes
    - batchB: a set of nodes which are neighbors of batchA
    - batchN: a set of negative sample nodes far away from batchA/batchB
    Notes:
    - batchA and batchB have the same size, and they are row-to-row paired in
      training (u and v in Eq (1) in the GraphSage paper).
    - batchN is randomly selected. The entire set is far from any node in
      batchA/batchB. There is a small chance that a node in batchN is close
      to a node in batchA/batchB.
    """

    batchA, batchB = zip(*edges)
    possible_negs = reduce ( np.setdiff1d
                           , ( neigh_dict.keys()
                             , batchA
                             , _get_neighbors(batchA, neigh_dict)
                             , batchB
                             , _get_neighbors(batchB, neigh_dict)
                             )
                           )
    batchN = np.random.choice ( possible_negs
                              , min(neg_size, len(possible_negs))
                              , replace=False
                              )
    batch_all = reduce(np.union1d, (batchA, batchB, batchN))

    # order does matter, in the model, use tf.gather on this
    dst2batchA = [np.where(batch_all == n)[0][0] for n in batchA]
    # order does matter, in the model, use tf.gather on this
    dst2batchB = [np.where(batch_all == n)[0][0] for n in batchB]
    # order does not matter, in the model, use tf.boolean_mask on this
    dst2batchN = np.any(np.stack([(batch_all == n) for n in batchN]), axis=0)

    minibatch_plain = build_batch_from_nodes ( num_layers
                                             , batch_all
                                             , neigh_dict
                                             , sample_size
                                             )

    MiniBatchFields = [ "src_nodes", "dif_mats", "dstsrc2srcs", "dstsrc2dsts"
                      , "dst2batchA", "dst2batchB", "dst2batchN" ]
    MiniBatch = collections.namedtuple ("MiniBatch", MiniBatchFields)

    return MiniBatch ( minibatch_plain.src_nodes
                     , minibatch_plain.dif_mats
                     , minibatch_plain.dstsrc2srcs
                     , minibatch_plain.dstsrc2dsts
                     , dst2batchA
                     , dst2batchB
                     , dst2batchN
                     )

def build_batch_from_nodes(num_layers, nodes, neigh_dict, sample_size):
    """
    This batch method is used directly in supervised mode and indirectly in
    unsupervised mode. It prepares auxiliary matrices for different layers
    according to the final output nodes.

    :param int num_layers: number of layers
    :param [int] nodes: node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param int sample_size: sample size
    :return namedtuple minibatch
        "src_nodes": node ids to retrieve from raw feature and feed to the first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """
    
    dst_nodes = [nodes]
    dstsrc2dsts = []
    dstsrc2srcs = []
    dif_mats = []

    for _ in range(num_layers):
        ds, d2d, d2s, dm = _compute_diffusion_matrix_mean ( dst_nodes[-1]
                                                          , neigh_dict
                                                          , sample_size
                                                          )
        dst_nodes.append(ds)
        dstsrc2dsts.append(d2d)
        dstsrc2srcs.append(d2s)
        dif_mats.append(dm)

    src_nodes = dst_nodes.pop()
    
    MiniBatchFields = ["src_nodes", "dif_mats", "dstsrc2srcs", "dstsrc2dsts"]
    MiniBatch = collections.namedtuple ("MiniBatch", MiniBatchFields)

    return MiniBatch(src_nodes, dif_mats, dstsrc2srcs, dstsrc2dsts)

################################################################
#                       Private Functions                      #
################################################################

def _compute_diffusion_matrix_mean(dst_nodes, neigh_dict, sample_size):
    neigh_mat = _sample_neighbors(dst_nodes, neigh_dict, sample_size)
    neigh_bool = np.any(neigh_mat.astype(np.bool), axis=0)

    # compute diffusion matrix
    neigh_mat_scoped = neigh_mat[:, neigh_bool]
    neigh_mat_scoped_sum = np.sum(neigh_mat_scoped, axis=1, keepdims=True)
    dif_mat = neigh_mat_scoped / neigh_mat_scoped_sum

    # compute dstsrc mappings
    src_nodes = np.arange(neigh_bool.size)[neigh_bool]
    dstsrc = np.union1d(dst_nodes, src_nodes)
    dstsrc2dst = np.any(np.stack([(dstsrc == n) for n in dst_nodes]), axis=0)
    dstsrc2src = np.any(np.stack([(dstsrc == n) for n in src_nodes]), axis=0)

    return dstsrc, dstsrc2dst, dstsrc2src, dif_mat

def _sample_neighbors(nodes, neigh_dict, sample_size):
    """
    return a sampled adjacency matrix from nodes to its neighbors
    """

    def sample(ns):
        return np.random.choice(ns, min(len(ns), sample_size), replace=False)

    def vectorize(ns):
        v = np.zeros(len(neigh_dict), dtype=np.float32)
        v[ns] = 1
        return v

    return np.stack([vectorize(sample(neigh_dict[n])) for n in nodes])

def _get_neighbors(nodes, neigh_dict):
    """
    return an array of neighbors of all nodes in the input
    """
    return reduce(np.union1d, [neigh_dict[n] for n in nodes])
