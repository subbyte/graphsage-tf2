#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import time
from itertools import islice
from collections import defaultdict, namedtuple
from sklearn.metrics import f1_score

from graphsage import GraphSage

#### NN parameters
SAMPLE_SIZE = 5
INTERNAL_DIM = 128
# number of layers, a positive integer
NUM_LAYERS = 2
#### training parameters
BATCH_SIZE = 256
TRAINING_STEPS = 100
LEARNING_RATE = 0.5

def build_batch(nodes, neigh_dict, sample_size):
    MiniBatchFields = ["dst", "neigh_dict", "sample_size"]
    MiniBatch = namedtuple ("MiniBatch", MiniBatchFields)
    return MiniBatch(np.array(nodes), neigh_dict, sample_size)

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats), dtype=np.float32)
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    adj_lists = {k: np.array(list(v)) for k, v in adj_lists.items()}
    
    return num_nodes, feat_data, labels, len(label_map), adj_lists

def run_cora():
    num_nodes, raw_features, labels, num_classes, neigh_dict = load_cora()

    graphsage = GraphSage(raw_features, INTERNAL_DIM, NUM_LAYERS, num_classes)

    all_nodes = np.random.permutation(num_nodes)
    train_nodes = all_nodes[:2048]
    test_nodes = all_nodes[2048:]

    # training
    def generate_training_minibatch(nodes_for_training, all_labels, batch_size):
        random.shuffle(nodes_for_training)
        while True:
            mini_batch_nodes = nodes_for_training[:batch_size]
            batch = build_batch(mini_batch_nodes, neigh_dict, SAMPLE_SIZE)
            labels = all_labels[mini_batch_nodes]
            yield (batch, labels)

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    minibatch_generator = generate_training_minibatch(train_nodes, labels, BATCH_SIZE)

    times = []
    for inputs, inputs_labels in islice(minibatch_generator, 0, TRAINING_STEPS):
        start_time = time.time()
        with tf.GradientTape() as tape:
            predicted = graphsage(inputs)
            loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)

        grads = tape.gradient(loss, graphsage.trainable_weights)
        optimizer.apply_gradients(zip(grads, graphsage.trainable_weights))
        end_time = time.time()
        times.append(end_time - start_time)
        print("Loss:", loss.numpy())

    # testing
    results = graphsage(build_batch(test_nodes, neigh_dict, SAMPLE_SIZE))
    score = f1_score(labels[test_nodes], results.numpy().argmax(axis=1), average="micro")
    print("Validation F1: ", score)
    print("Average batch time: ", np.mean(times))

if __name__ == "__main__":
    run_cora()
