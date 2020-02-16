#!/usr/bin/env python3

import tensorflow as tf
from meanaggregate import compute_auxiliary_matrices as cam

init_fn = tf.keras.initializers.GlorotUniform

class GraphSage(tf.keras.Model):
    """
    embed nodes and classify them with labels
    """

    def __init__(self, raw_features, internal_dim, num_layers, num_classes):

        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'

        super().__init__()

        self.input_layer = RawFeature(raw_features)
        self.num_layers = num_layers

        self.seq_layers = []
        for i in range (1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            input_dim = internal_dim if i > 1 else raw_features.shape[-1]
            aggregator_layer = MeanAggregator(input_dim, internal_dim, name=layer_name)
            self.seq_layers.append(aggregator_layer)

        self.classifier = tf.keras.layers.Dense ( num_classes
                                                , activation = tf.nn.softmax
                                                , use_bias = False
                                                , kernel_initializer = init_fn
                                                , name = "classifier"
                                                )

    def call(self, minibatch):
        """
        :param [node] nodes: target nodes for embedding
        """
        # prepare diffusion and helper matrices/mappings
        src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats = cam ( minibatch.dst
                                                            , minibatch.neigh_dict
                                                            , self.num_layers
                                                            , minibatch.sample_size
                                                            )

        # forward propagation
        x = self.input_layer(tf.squeeze(src_nodes))

        for aggregator_layer in self.seq_layers:
            x = aggregator_layer ( x
                                 , dstsrc2dsts.pop()
                                 , dstsrc2srcs.pop()
                                 , dif_mats.pop()
                                 )

        y = self.classifier(x)

        return y

################################################################
#                     Custom Layers (Private)                  #
################################################################

class RawFeature(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        """
        :param ndarray((#(node), #(feature))) features: a matrix, each row is feature for a node
        """
        super().__init__(**kwargs)
        self.features = tf.constant(features)
        
    def call(self, nodes):
        """
        :param [int] nodes: node ids
        """
        return tf.gather(self.features, nodes)

class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(**kwargs)

        kwargs["name"]

        self.w = self.add_weight( name = kwargs["name"] + "_weight"
                                , shape = (src_dim*2, dst_dim)
                                , dtype = tf.float32
                                , initializer = init_fn
                                , trainable = True
                                )
    
    def call(self, dstsrc_features, dstsrc2dst, dstsrc2src, dif_mat):
        """
        :param tensor dstsrc_features: the embedding from the previous layer
        :param tensor dstsrc2dst: 1d index mapping
        :param tensor dstsrc2src: 1d index mapping
        :param tensor dif_mat: 2d diffusion matrix (prepraed by minibatch generator)
        """
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        aggregated_features = tf.matmul(dif_mat, src_features)
        concatenated_features = tf.concat([aggregated_features, dst_features], 1)
        x = tf.matmul(concatenated_features, self.w)
        return tf.nn.relu(x)
