# TensorFlow 2 Implementation of GraphSage

GraphSAGE: https://github.com/williamleif/GraphSAGE

This is a TensorFlow 2 implementation of graphsage-simple:
- original PyTorch version: https://github.com/williamleif/graphsage-simple
- version with Python 3 and PyTorch 1.3: https://github.com/subbyte/graphsage-simple

### Simple Implementation for Studying Purpose
- It includes training/testing on the cora dataset.
- It only run in the supervised mode.
- It only has MeanAggregator.
- It only has two layers hard-coded.
- It only has CPU support.

### Prilimary Performance Evaluation

Given the same size of minibatch (256 units), this TensorFlow version is 25% faster on CPU. The average batch time in training with 32-core CPU:
- graphsage-simple (PyTorch): 0.038063554763793944s
- graphsage-tf2 (TensorFlow): 0.02766340970993042s
