#!/usr/bin/env python3

"""
This module computes minibatch (data and auxiliary matrices) for mean aggregator

requirement: neigh_dict is a BIDIRECTIONAL adjacency matrix in dict
"""

import numpy as np
import collections
import random

def build_batch_from_nodes(num_layers, nodes, neigh_dict, sample_size):
    """
    :param int num_layers: number of layers
    :param [int] nodes: node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param int sample_size: sample size
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

    for _ in range(num_layers):
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
