# Yet Another GraphSage Implementation in TensorFlow2

GraphSAGE (original implementation): https://github.com/williamleif/GraphSAGE

### Simple Implementation for Studying Purpose

This is another GraphSAGE implementation
- Simplifying things for studying purpose
    - It includes the toy cora dataset
    - It only has MeanAggregator
    - It only has CPU support
- Number of layers as a hyperparameter
- TensorFlow 2
- Python 3

### How to Run 

#### Supervised Version
It requires `python3` with both `tensorflow` and `scikit-learn` packages.
```
./supervised.py
```

#### Unsupervised Version (Under Development)
```
./unsupervised.py
```

### Design Choices
1. Computing diffusion matrix and bitmask for node feature and neighbor concatenation outside TensorFlow (in minibatch.py) since this is faster. Tried to write this part inside tensorflow but got 50x slower (branch #internaldiffusion) not using tf.function. It seems too complicated to get rid of python dict and use only matrices and tf.function, which I haven't tested.
2. `fit` and `fit_generator` in `tf.keras.Model` do not support more than one arguments, so I manually write the training loop with `tf.GradientTape()`.

### Prilimary Performance Evaluation

This implementation is 35% faster than the PyTorch version. Given the same size of minibatch (256 units), the average batch time in training with 32-core CPU:
- graphsage-simple (PyTorch, [updated version](https://github.com/subbyte/graphsage-simple)): 0.038063554763793944s
- graphsage-tf2 (TensorFlow): 0.02408278703689575s
