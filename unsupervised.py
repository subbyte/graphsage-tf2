#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import time
from itertools import islice
from sklearn.metrics import f1_score

from dataloader.cora import load_cora
from dataloader.ppi import load_ppi
from minibatch import build_batch_from_edges as build_batch
from graphsage import GraphSageUnsupervised as GraphSage
from config import ENABLE_UNKNOWN_OP

#### NN parameters
SAMPLE_SIZES = [25, 10]
INTERNAL_DIM = 128
NEG_WEIGHT = 1.0
#### training parameters
BATCH_SIZE = 512
NEG_SIZE = 20
TRAINING_STEPS = 100
LEARNING_RATE = 0.00001

def generate_training_minibatch(adj_mat_dict, batch_size, sample_sizes, neg_size):
    edges = np.array([(k, v) for k in adj_mat_dict for v in adj_mat_dict[k]])
    nodes = np.array(list(adj_mat_dict.keys()))
    while True:
        mini_batch_edges = edges[np.random.randint(edges.shape[0], size = batch_size), :]
        batch = build_batch(mini_batch_edges, nodes, adj_mat_dict, sample_sizes, neg_size)
        yield batch

def run_cora():
    # num_nodes, raw_features, labels, num_classes, neigh_dict = load_cora()
    num_nodes, raw_features, neigh_dict = load_ppi()

    if ENABLE_UNKNOWN_OP:
        # /graphsage/unsupervised_train.py, line 139
        raw_features = np.vstack([raw_features, np.zeros((raw_features.shape[1],))])

    minibatch_generator = generate_training_minibatch ( neigh_dict
                                                      , BATCH_SIZE
                                                      , SAMPLE_SIZES
                                                      , NEG_SIZE
                                                      )

    graphsage = GraphSage(raw_features, INTERNAL_DIM, len(SAMPLE_SIZES), NEG_WEIGHT)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # training
    times = []
    for minibatch in islice(minibatch_generator, 0, TRAINING_STEPS):
        start_time = time.time()
        with tf.GradientTape() as tape:
            loss = graphsage(minibatch)
        grads = tape.gradient(loss, graphsage.trainable_weights)
        optimizer.apply_gradients(zip(grads, graphsage.trainable_weights))
        end_time = time.time()
        times.append(end_time - start_time)
        print("Loss:", loss.numpy())
    print("Average batch time: ", np.mean(times))

if __name__ == "__main__":
    run_cora()
