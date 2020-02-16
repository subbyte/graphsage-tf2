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
    neigh_mat = _sample_neighbors(dst_nodes, neigh_dict, sample_size, max_node_id)
    neigh_bool = np.any(neigh_mat.astype(np.bool), axis=0)

    # compute diffusion matrix
    neigh_mat_scoped = neigh_mat[:, neigh_bool]
    neigh_mat_scoped_sum = np.sum(neigh_mat_scoped, axis=1, keepdims=True)
    dif_mat = neigh_mat_scoped / neigh_mat_scoped_sum

    # compute dstsrc mappings
    src_nodes = np.arange(neigh_bool.size)[neigh_bool]
    dstsrc = np.union1d(dst_nodes, src_nodes)
    dstsrc2src = np.any(np.stack([(dstsrc == n) for n in src_nodes]), axis=0)
    dstsrc2dst = np.any(np.stack([(dstsrc == n) for n in dst_nodes]), axis=0)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat

def _sample_neighbors(nodes, neigh_dict, sample_size, max_node_id):
    """
    return a sampled adjacency matrix from nodes to its neighbors
    """

    def sample(ns):
        return np.random.choice(ns, min(len(ns), sample_size), replace=False)

    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    return np.stack([vectorize(sample(neigh_dict[n])) for n in nodes])
