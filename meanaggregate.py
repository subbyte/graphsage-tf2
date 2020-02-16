#!/usr/bin/env python3

import tensorflow as tf

def compute_auxiliary_matrices(dst, neigh_dict, num_layers, sample_size):
    """
    :param [int] dst: node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param int num_layers: number of layers
    :param int sample_size: sample size
    :return auxiliary matrices
        "src_nodes": node ids to retrieve from raw feature and feed to the first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """
    
    dst_nodes = [dst]
    dstsrc2dsts = []
    dstsrc2srcs = []
    dif_mats = []

    for _ in range(num_layers):
        ds, d2s, d2d, dm = _compute_diffusion_matrix ( dst_nodes[-1]
                                                     , neigh_dict
                                                     , sample_size
                                                     )
        dst_nodes.append(ds)
        dstsrc2srcs.append(d2s)
        dstsrc2dsts.append(d2d)
        dif_mats.append(dm)

    src_nodes = dst_nodes.pop()
    
    return src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats

################################################################
#                       Private Functions                      #
################################################################

def _compute_diffusion_matrix(dst, neigh_dict, sample_size):
    """
    return diffusion matrix and node mappings
    """
    assert len(dst.shape) == 1
    assert dst.dtype == tf.int64
    assert sample_size > 0

    # compute full adj matrix, which has empty columns
    row_len = tf.math.add(_max_neighbor(neigh_dict), 1)
    adj_mat_full = tf.map_fn ( lambda x: _vectorize ( _sample ( neigh_dict[int(x)]
                                                              , sample_size )
                                                    , row_len )
                             , dst
                             , dtype=tf.float32
                             )
    nonzero_cols_mask = tf.reduce_any(tf.cast(adj_mat_full, tf.bool), axis=0)

    # compute diffusion matrix
    adj_mat = tf.boolean_mask(adj_mat_full, nonzero_cols_mask, axis=1)
    adj_mat_sum = tf.reduce_sum(adj_mat, axis=1, keepdims=True)
    dif_mat = adj_mat / adj_mat_sum

    # compute dstsrc mappings
    src = tf.boolean_mask(tf.range(row_len, dtype=tf.int64), nonzero_cols_mask)
    dstsrc = tf.unique(tf.concat([dst, src], 0)).y
    dstsrc2src = tf.map_fn(lambda x: tf.where(tf.equal(dstsrc, x))[0][0], src)
    dstsrc2dst = tf.map_fn(lambda x: tf.where(tf.equal(dstsrc, x))[0][0], dst)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat

def _max_neighbor(neigh_dict):
    """
    return the maximum neighbor in neigh_dict
    """
    return int(tf.math.reduce_max(tf.concat(list(neigh_dict.values()), 0)))

def _sample(tensor, sample_size):
    """
    sample a number of items along 0-axis of the input tensor

    uniq = True
    if sample_size > all items, output all items
    """
    return tf.random.shuffle(tensor)[:min(sample_size, tensor.shape[0])]

def _vectorize(neigh_nodes, max_len):
    """
    build a line of the float32 adj matrix
    """
    assert len(neigh_nodes.shape) == 1
    assert neigh_nodes.dtype == tf.int64

    adj_line = tf.SparseTensor ( tf.expand_dims(neigh_nodes, -1)
                               , tf.ones_like(neigh_nodes, dtype=tf.float32)
                               , [max_len]
                               )
    return tf.sparse.to_dense(tf.sparse.reorder(adj_line))
