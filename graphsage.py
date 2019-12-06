#!/usr/bin/env python3

import tensorflow as tf

init_fn = tf.keras.initializers.GlorotUniform

class GraphSage(tf.keras.Model):
    """
    compute embedding for target nodes
    """

    def __init__(self, raw_features, internal_dim, num_classes):
        super().__init__()
        self.level_0 = RawFeature(raw_features)
        self.level_1 = MeanAggregator(raw_features.shape[-1], internal_dim, name="agg_lv1")
        self.level_2 = MeanAggregator(internal_dim, internal_dim, name="agg_lv2")
        self.classifier = tf.keras.layers.Dense( num_classes
                                               , activation = tf.nn.softmax
                                               , use_bias = False
                                               , kernel_initializer = init_fn
                                               , name = "classifier"
                                               )

    def call(self, minibatch):
        """
        :param [node] nodes: target nodes for embedding
        """
        x = self.level_0(tf.squeeze(minibatch["src_nodes"]))
        x = self.level_1(x, minibatch["dstsrc2dst_1"], minibatch["dstsrc2src_1"], minibatch["dif_mat_1"])
        x = self.level_2(x, minibatch["dstsrc2dst_0"], minibatch["dstsrc2src_0"], minibatch["dif_mat_0"])
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
    """
    Usage (forward):
    1. agg = MeanAggregator(adj_mat, dst_dim)
    2. dstsrc = agg.get_associated_nodes(dst_nodes) 
       # dstsrc is the dst_nodes for the previous layer
       # this step will also prepare the diffusion matrix
    3. embeddings = agg(features)
       # features is the embedding from the previous layer
       # there are #(dstsrc) features
       # there are #(dst_nodes) embedding outputs
    """
    
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
        dst_features = tf.boolean_mask(dstsrc_features, dstsrc2dst)
        src_features = tf.boolean_mask(dstsrc_features, dstsrc2src)
        aggregated_features = tf.matmul(dif_mat, src_features)
        concatenated_features = tf.concat([aggregated_features, dst_features], 1)
        x = tf.matmul(concatenated_features, self.w)
        return tf.nn.relu(x)
