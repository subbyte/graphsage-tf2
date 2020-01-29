# TensorFlow 2 Implementation of GraphSage

GraphSAGE: https://github.com/williamleif/GraphSAGE

This is a TensorFlow 2 implementation of graphsage-simple in Python 3:
- original PyTorch version: https://github.com/williamleif/graphsage-simple
- version with Python 3 and PyTorch 1.3: https://github.com/subbyte/graphsage-simple

### How to Run 

It requires both `tensorflow` and `scikit-learn` packages. If you are using TensorFlow docker containers, please use the Python 3 containers `*-py3` and install `scikit-learn` before running this.

`python3 supervised.py`

### Simple Implementation for Studying Purpose
- It includes training/testing on the cora dataset.
- It only run in the supervised mode.
- It only has MeanAggregator.
- It only has two layers hard-coded.
- It only has CPU support.

### Design Choices
1. Computing diffusion matrix and bitmask for node feature and neighbor concatenation outside TensorFlow (in minibatch.py) since I do not find a good way to sample neighbors in an adjancy matrix within `tf.function`. If possible, the code can be simpler without passing more than one argument tensors to the forward pass.
2. `fit` and `fit_generator` in `tf.keras.Model` do not support more than one arguments, so I manually write the training loop with `tf.GradientTape()`.

### Prilimary Performance Evaluation

Given the same size of minibatch (256 units), this TensorFlow version is 25% faster on CPU. The average batch time in training with 32-core CPU:
- graphsage-simple (PyTorch): 0.038063554763793944s
- graphsage-tf2 (TensorFlow): 0.02766340970993042s
