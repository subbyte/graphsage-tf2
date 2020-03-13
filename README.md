# Yet Another GraphSage Implementation in TensorFlow2

GraphSAGE (original implementation): https://github.com/williamleif/GraphSAGE

### Simple Implementation for Studying Purpose

This is another GraphSAGE implementation
- Simplifying things for studying purpose
    - MeanAggregator only
    - CPU runtime only
- Python 3
- TensorFlow 2

### How to Run 

#### Supervised Version
It requires `python3` with both `tensorflow` and `scikit-learn` packages.
```
./supervised.py
```

#### Unsupervised Version
```
./unsupervised.py
```

### Design Choices
1. Computing diffusion matrix and bitmask for node feature and neighbor concatenation outside TensorFlow (in minibatch.py) since this is faster. Tried to write this part inside tensorflow but got 50x slower (branch #internaldiffusion) not using tf.function. It seems too complicated to get rid of python dict and use only matrices and tf.function, which I haven't tested.
2. `fit` and `fit_generator` in `tf.keras.Model` do not support more than one arguments, so I manually write the training loop with `tf.GradientTape()`.

### Prilimary Performance Evaluation (CPU)

The supervised part is 35% faster than [graphsage-simple](https://github.com/williamleif/graphsage-simple/) (PyTorch version of GraphSage)
- cora dataset, minibatch size 256 units, average batch time on a 32-vcore testbed:
- graphsage-simple [updated version](https://github.com/subbyte/graphsage-simple): 0.03806s
- graphsage-tf2 (this one): 0.02408s

This unsupervised part is much slower than GraphSage [original implementation](https://github.com/williamleif/GraphSAGE/) and cannot parallelize well with multi-core CPU. It is under investigation.
