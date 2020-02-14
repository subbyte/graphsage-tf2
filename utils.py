#!/usr/bin/env python3

import tensorflow as tf

def compute_diffusion_matrix(dst, neigh_dict, sample_size):
    """
    return diffusion matrix and node mappings
    """
    assert len(dst.shape) == 1
    assert dst.dtype == tf.int64
    assert sample_size > 0

    # compute adj matrix with zero columns
    row_len = max_neighbor(neigh_dict)
    adj_mat_full = tf.map_fn ( lambda x: vectorize ( sample ( neigh_dict[int(x)]
                                                            , sample_size )
                                                   , row_len )
                             , dst )
    nonzero_cols = tf.reduce_any(tf.cast(adj_mat_full, tf.bool), axis=0)

    # compute diffusion matrix
    adj_mat = tf.boolean_mask(adj_mat_full, nonzero_cols, axis=1)
    adj_mat_sum = tf.reduce_sum(adj_mat, axis=1, keepdims=True)
    dif_mat = adj_mat / adj_mat_sum

    # compute dstsrc mappings
    src = tf.boolean_mask(tf.range(row_len, dtype=tf.int64), nonzero_cols)
    dstsrc = tf.unique(tf.concat([dst, src], 0)).y
    dstsrc2src = tf.map_fn(lambda x: tf.where(tf.equal(dstsrc, x))[0][0], src)
    dstsrc2dst = tf.map_fn(lambda x: tf.where(tf.equal(dstsrc, x))[0][0], dst)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat

def sample(tensor, sample_size):
    """
    sample a number of items along 0-axis of the input tensor

    uniq = True
    if sample_size > all items, output all items
    """
    return tf.random.shuffle(tensor)[:min(sample_size, tensor.shape()[0])]

def max_neighbor(neigh_dict):
    """
    return the maximum neighbor in neigh_dict
    """
    return int(tf.math.reduce_max(tf.concat(list(neigh_dict.values()), 0)))

def vectorize(neigh_nodes, max_len):
    """
    build a line of the float32 adj matrix
    """
    assert len(neigh_nodes.shape) == 1
    assert neigh_nodes.dtype == tf.int64

    adj_line = tf.SparseTensor ( tf.expand_dims(neigh_nodes, -1)
                               , tf.ones_like(neigh_nodes, dtype=tf.float32)
                               , [max_len]
                               )
    return tf.sparse.to_dense(adj_line)
