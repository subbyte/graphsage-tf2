#!/usr/bin/env python3

import numpy as np
import random

class Batch:
    def __init__ (self, src_nodes, dif_mat, dstsrc2src, dstsrc2dst):
        self.src_nodes = src_nodes
        self.dif_mats = dif_mat
        self.dstsrc2srcs = dstsrc2src
        self.dstsrc2dsts = dstsrc2dst

def build_batch(num_layers, nodes, neigh_dict, sample_size):
    """
    :param int num_layers: number of layers
    :param [int] nodes: node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param int sample_size: sample size
    :return dict minibatch
        "src_nodes": node ids to retrieve from raw feature and feed to the first layer
        "dstsrc2dst": list of dstsrc2dst matrices from last to first layer
        "dstsrc2src": list of dstsrc2src matrices from last to first layer
        "dif_mat": list of dif_mat matrices from last to first layer
    """
    
    dst_nodes = [nodes]
    dstsrc2dst = []
    dstsrc2src = []
    dif_mat = []

    for _ in range(num_layers):
        ds, d2d, d2s, dm = compute_diffusion_matrix_mean(dst_nodes[-1], neigh_dict, sample_size)
        dst_nodes.append(ds)
        dstsrc2dst.append(d2d)
        dstsrc2src.append(d2s)
        dif_mat.append(dm)

    src_nodes = dst_nodes.pop()
    
    return Batch(src_nodes, dif_mat, dstsrc2src, dstsrc2dst)

################################################################
#                       Private Functions                      #
################################################################

def compute_diffusion_matrix_mean(dst_nodes, neigh_dict, sample_size):
    neigh_mat = sample_neighbors(dst_nodes, neigh_dict, sample_size)
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

def sample_neighbors(nodes, neigh_dict, sample_size):
    """
    return a sampled adjacency matrix from nodes to its neighbors
    """

    def sample(ns):
        return random.sample(ns, min(len(ns), sample_size))

    def vectorize(ns):
        v = np.zeros(len(neigh_dict), dtype=np.float32)
        v[ns] = 1
        return v

    return np.stack([vectorize(sample(neigh_dict[n])) for n in nodes])
