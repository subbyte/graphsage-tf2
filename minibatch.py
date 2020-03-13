#!/usr/bin/env python3

"""
This module computes minibatch (data and auxiliary matrices) for mean aggregator

requirement: neigh_dict is a BIDIRECTIONAL adjacency matrix in dict
"""

import numpy as np
import collections
from functools import reduce

def build_batch_from_edges(edges, nodes, neigh_dict, sample_sizes, neg_size):
    """
    This batch method is used for unsupervised mode. First, it prepares
    auxiliary matrices for the combination of neighbor nodes (read from edges)
    and negative sample nodes. Second, it provides mappings to filter the
    results into three portions for use in the unsupervised loss function.

    :param array([(int, int)]) edges: edge with node ids
    :param array([int]) nodes: all node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param [sample_size]: sample sizes for each layer, lens is the number of layers
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

    batchA, batchB = edges.transpose()

    possible_negs = reduce ( np.setdiff1d
                           , ( nodes
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

    # np.unique sorts the return, required by the following np.searchsorted
    batch_all = np.unique(np.concatenate((batchA, batchB, batchN)))
    # order does matter, in the model, use tf.gather on this
    dst2batchA = np.searchsorted(batch_all, batchA)
    # order does matter, in the model, use tf.gather on this
    dst2batchB = np.searchsorted(batch_all, batchB)
    # order does not matter, in the model, use tf.boolean_mask on this
    dst2batchN = np.in1d(batch_all, batchN)

    minibatch_plain = build_batch_from_nodes ( batch_all
                                             , neigh_dict
                                             , sample_sizes
                                             )

    MiniBatchFields = [ "src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"
                      , "dst2batchA", "dst2batchB", "dst2batchN" ]
    MiniBatch = collections.namedtuple ("MiniBatch", MiniBatchFields)

    return MiniBatch ( minibatch_plain.src_nodes
                     , minibatch_plain.dstsrc2srcs
                     , minibatch_plain.dstsrc2dsts
                     , minibatch_plain.dif_mats
                     , dst2batchA
                     , dst2batchB
                     , dst2batchN
                     )

def build_batch_from_nodes(nodes, neigh_dict, sample_sizes):
    """
    :param [int] nodes: node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param [sample_size]: sample sizes for each layer, lens is the number of layers
    :return namedtuple minibatch
        "src_nodes": node ids to retrieve from raw feature and feed to the first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """
    
    dst_nodes = [nodes]
    dstsrc2dsts = []
    dstsrc2srcs = []
    dif_mats = []

    max_node_id = max(list(neigh_dict.keys()))

    for sample_size in reversed(sample_sizes):
        ds, d2s, d2d, dm = _compute_diffusion_matrix ( dst_nodes[-1]
                                                     , neigh_dict
                                                     , sample_size
                                                     , max_node_id
                                                     )
        dst_nodes.append(ds)
        dstsrc2srcs.append(d2s)
        dstsrc2dsts.append(d2d)
        dif_mats.append(dm)

    src_nodes = dst_nodes.pop()
    
    MiniBatchFields = ["src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"]
    MiniBatch = collections.namedtuple ("MiniBatch", MiniBatchFields)

    return MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats)

################################################################
#                       Private Functions                      #
################################################################

def _compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size, max_node_id):

    def sample(ns):
        return np.random.choice(ns, min(len(ns), sample_size), replace=False)

    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    # sample neighbors
    adj_mat_full = np.stack([vectorize(sample(neigh_dict[n])) for n in dst_nodes])
    nonzero_cols_mask = np.any(adj_mat_full.astype(np.bool), axis=0)

    # compute diffusion matrix
    adj_mat = adj_mat_full[:, nonzero_cols_mask]
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)
    dif_mat = adj_mat / adj_mat_sum

    # compute dstsrc mappings
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
    # np.union1d automatic sorts the return, which is required for np.searchsorted
    dstsrc = np.union1d(dst_nodes, src_nodes)
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat

def _get_neighbors(nodes, neigh_dict):
    """
    return an array of neighbors of all nodes in the input
    """
    return np.unique(np.concatenate([neigh_dict[n] for n in nodes]))
