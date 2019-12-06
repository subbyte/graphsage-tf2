# TensorFlow 2 Implementation of GraphSage

This is an implementation of graphsage-simple:
- original PyTorch version: https://github.com/williamleif/graphsage-simple
- version with Python 3 and PyTorch 1.3: https://github.com/subbyte/graphsage-simple

It includes training/testing on the cora dataset.

Given the same size of minibatch (256 units), this TensorFlow version is 25% faster on CPU. The average batch time in training with 32-core CPU:
- graphsage-simple (PyTorch): 0.038063554763793944s
- this version (TensorFlow 2): 0.02766340970993042s
