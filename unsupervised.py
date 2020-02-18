#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import time
from itertools import islice
from sklearn.metrics import f1_score

from dataloader import load_cora
from minibatch import build_batch_from_edges as build_batch
from graphsage import GraphSageUnsupervised as GraphSage

#### NN parameters
SAMPLE_SIZE = 10
INTERNAL_DIM = 128
NUM_LAYERS = 2
NEG_WEIGHT = 1.0
#### training parameters
BATCH_SIZE = 512
NEG_SIZE = 20
TRAINING_STEPS = 1000
LEARNING_RATE = 0.001

def generate_training_minibatch(adj_mat_dict, batch_size, num_layer, sample_size, neg_size):
    edges = [(k, v) for k in adj_mat_dict for v in adj_mat_dict[k]]
    while True:
        mini_batch_edges = random.sample(edges, batch_size)
        batch = build_batch(num_layer, mini_batch_edges, adj_mat_dict, sample_size, neg_size)
        yield batch

def run_cora():
    num_nodes, raw_features, labels, num_classes, neigh_dict = load_cora()

    minibatch_generator = generate_training_minibatch ( neigh_dict
                                                      , BATCH_SIZE
                                                      , NUM_LAYERS
                                                      , SAMPLE_SIZE
                                                      , NEG_SIZE
                                                      )

    graphsage = GraphSage(raw_features, INTERNAL_DIM, NUM_LAYERS, NEG_WEIGHT)

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
