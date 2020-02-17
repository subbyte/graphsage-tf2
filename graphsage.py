#!/usr/bin/env python3

import tensorflow as tf

init_fn = tf.keras.initializers.GlorotUniform

class GraphSageBase(tf.keras.Model):
    """
    GraphSage base model without last layer
    """

    def __init__(self, raw_features, internal_dim, num_layers, last_has_activ):

        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'

        super().__init__()

        self.input_layer = RawFeature(raw_features, name="raw_feature_layer")

        self.seq_layers = []
        for i in range (1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            input_dim = internal_dim if i > 1 else raw_features.shape[-1]
            has_activ = last_has_activ if i == num_layers else True
            aggregator_layer = MeanAggregator ( input_dim
                                              , internal_dim
                                              , name=layer_name
                                              , activ = has_activ
                                              )
            self.seq_layers.append(aggregator_layer)

    def call(self, minibatch):
        """
        :param [node] nodes: target nodes for embedding
        """
        x = self.input_layer(tf.squeeze(minibatch.src_nodes))
        for aggregator_layer in self.seq_layers:
            x = aggregator_layer ( x
                                 , minibatch.dstsrc2srcs.pop()
                                 , minibatch.dstsrc2dsts.pop()
                                 , minibatch.dif_mats.pop()
                                 )
        return x

class GraphSageSupervised(GraphSageBase):
    def __init__(self, raw_features, internal_dim, num_layers, num_classes):
        super().__init__(raw_features, internal_dim, num_layers, True)
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
        return self.classifier( super().call(minibatch) )

class GraphSageUnsupervised(GraphSageBase):
    def __init__(self, raw_features, internal_dim, num_layers, neg_weight):
        super().__init__(raw_features, internal_dim, num_layers, False)
        self.trainloss = UnsupervisedTrainLoss(neg_weight)

    def call(self, minibatch):
        embeddingABN = super().call(minibatch)
        embeddingA = tf.gather(embeddingABN, minibatch.dst2batchA)
        embeddingB = tf.gather(embeddingABN, minibatch.dst2batchB)
        embeddingN = tf.boolean_mask(embeddingABN, minibatch.dst2batchN)
        return self.trainloss(embeddingA, embeddingB, embeddingN)

################################################################
#                         Custom Layers                        #
################################################################

class RawFeature(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        """
        :param ndarray((#(node), #(feature))) features: a matrix, each row is feature for a node
        """
        super().__init__(trainable=False, **kwargs)
        self.features = tf.constant(features)
        
    def call(self, nodes):
        """
        :param [int] nodes: node ids
        """
        return tf.gather(self.features, nodes)

class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight( name = kwargs["name"] + "_weight"
                                , shape = (src_dim*2, dst_dim)
                                , dtype = tf.float32
                                , initializer = init_fn
                                , trainable = True
                                )
    
    def call(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat):
        """
        :param tensor dstsrc_features: the embedding from the previous layer
        :param tensor dstsrc2dst: 1d index mapping (prepraed by minibatch generator)
        :param tensor dstsrc2src: 1d index mapping (prepraed by minibatch generator)
        :param tensor dif_mat: 2d diffusion matrix (prepraed by minibatch generator)
        """
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        aggregated_features = tf.matmul(dif_mat, src_features)
        concatenated_features = tf.concat([aggregated_features, dst_features], 1)
        x = tf.matmul(concatenated_features, self.w)
        return self.activ_fn(x)

################################################################
#                   Custom Layers (Unsupervised)               #
################################################################

class UnsupervisedTrainLoss(tf.keras.layers.Layer):
    """
    Implement the loss function of unsupervised training as a layer
    """

    def __init__(self, neg_weight, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.neg_weight = neg_weight

    def call(self, embeddingA, embeddingB, embeddingN):
        """
        compute and return the loss based on Eq (1) in the GraphSage paper

        :param 1d-tensor embeddingA: embedding of a list of nodes
        :param 1d-tensor embeddingB: embedding of a list of neighbor nodes
                                     pairwise to embeddingA
        :param 1d-tensor embeddingN: embedding of a list of non-neighbor nodes
                                     (negative samples) to embeddingA
        """
        # positive affinity: pair-wise calculation
        pos_affinity = tf.reduce_sum ( tf.math.multiply ( embeddingA, embeddingB ) )
        # negative affinity: enumeration of all combinations of (embeddingA, embeddingN)
        neg_affinity = tf.matmul ( embeddingA, tf.transpose ( embeddingN ) )

        pos_xent = tf.nn.sigmoid_cross_entropy_with_logits ( tf.ones_like(pos_affinity)
                                                           , pos_affinity
                                                           , "positive_xent" )
        neg_xent = tf.nn.sigmoid_cross_entropy_with_logits ( tf.zeros_like(neg_affinity)
                                                           , neg_affinity
                                                           , "negative_xent" )
        return tf.reduce_sum(pos_xent) + self.neg_weight * tf.reduce_sum(neg_xent)
