from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
#try:
#    import special_grads
#except KeyError as e:
#    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e, file=sys.stderr)

from tensorflow.python.platform import flags
from utils import conv_block, fc, max_pool, lrn, dropout
from utils import xent, kd

FLAGS = flags.FLAGS

class MASF:
    def __init__(self, num_classes, batch_size=128):
        """ Call construct_model_*() after initializing MASF"""
        self.forward = self.forward_fc
        self.construct_weights = self.construct_fc_weights
        self.loss_func = self.softmax
        #self.loss_func = self.additive_angular_margin_softmax
        self.num_classes = num_classes
        self.KEEP_PROB = 1.0
        self.batch_size = batch_size


    def construct_model_train(self, prefix='metatrain_'):
        # a: meta-train for inner update, b: meta-test for meta loss
        self.input = tf.placeholder(tf.float32)
        self.label = tf.placeholder(tf.float32)

        meta_sample_num = (FLAGS.meta_batch_size /2) * 2

        self.inner_lr = FLAGS.inner_lr
        self.outer_lr = FLAGS.outer_lr
        self.clip_value = FLAGS.gradients_clip_value
        self.KEEP_PROB = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                print('weights already defined')
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            def task_metalearn(inp, reuse=True):
                # Function to perform meta learning update """
                #inputa0, labela0, inputa1, labela1, inputb, labelb = inp
                input, label = inp

                # Obtaining the conventional task loss on meta-train
                _, output = self.forward(input, weights, reuse=reuse, is_training=True)
                output, loss = self.loss_func(output, label, is_training=True, reuse_variables=tf.AUTO_REUSE)
                accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(output), 1), tf.argmax(label, 1)) #this accuracy already gathers batch size

                task_output = [loss, accuracy]
                return task_output

            self.global_step = tf.Variable(0, trainable=False)

            input_tensors = (self.input, self.label)
            result = task_metalearn(inp=input_tensors)
            self.raw_loss, accuracy = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.loss = avg_loss = tf.reduce_mean(self.raw_loss)
            self.task_train_op = tf.train.AdamOptimizer(learning_rate=self.outer_lr).minimize(self.loss, global_step=self.global_step)

            self.accuracy = accuracy * 100.

        ## Summaries
        tf.summary.scalar(prefix+'loss', self.loss)
        tf.summary.scalar(prefix+'accuracy', self.accuracy)


    def construct_model_predict(self, prefix='predict'):

        self.test_input = tf.placeholder(tf.float32)

        with tf.variable_scope('model') as testing_scope:
            self.weights = weights = self.construct_fc_weights()
            testing_scope.reuse_variables()

            embeddings, _= self.forward(self.test_input, weights)
 
        self.embeddings = embeddings


    def construct_fc_weights(self):

        weights = {}
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

        with tf.variable_scope('dense1') as scope:
            weights['dense1_weights'] = tf.get_variable('weights', shape=[512, 512], 
                               initializer=fc_initializer)
            weights['dense1_biases'] = tf.get_variable('biases', [512],
                               initializer=fc_initializer)

        with tf.variable_scope('dense2') as scope:
            weights['dense2_weights'] = tf.get_variable('weights', shape=[512, 512],
                               initializer=fc_initializer)
            weights['dense2_biases'] = tf.get_variable('biases', [512],
                               initializer=fc_initializer)

        with tf.variable_scope('dense3') as scope:
            weights['dense3_weights'] = tf.get_variable('weights', shape=[512, 512],
                               initializer=fc_initializer)
            weights['dense3_biases'] = tf.get_variable('biases', [512], 
                               initializer=fc_initializer)

        return weights


    def forward_fc(self, inp, weights, reuse=False, is_training=False):
        # reuse is for the normalization parameters.
        x = tf.reshape(inp, [-1,512])
        dense1 = fc(x, weights['dense1_weights'], weights['dense1_biases'], activation=None)
        bn1 = tf.layers.batch_normalization(dense1, momentum=0.99, training=is_training,  
                                            name='bn1', reuse=tf.AUTO_REUSE)
        relu1 = tf.nn.relu(bn1)
        dropout1 = dropout(relu1, self.KEEP_PROB)

        dense2 = fc(dropout1, weights['dense2_weights'], weights['dense2_biases'], activation=None)
        bn2 = tf.layers.batch_normalization(dense2, momentum=0.99, training=is_training,  
                                            name='bn2', reuse=tf.AUTO_REUSE)
        relu2 = tf.nn.relu(bn2)
        dropout2 = dropout(relu2, self.KEEP_PROB)

        dense3 = fc(dropout2, weights['dense3_weights'], weights['dense3_biases'], activation=None)
        bn3 = tf.layers.batch_normalization(dense3, momentum=0.99, training=is_training,  
                                            name='bn3', reuse=tf.AUTO_REUSE)
        relu3 = tf.nn.relu(bn3)
        if self.loss_func == self.additive_angular_margin_softmax:
            return dense2, bn3  # last_layer_linear for angular softmax
        elif self.loss_func == self.softmax:
            return dense2, relu3


    def additive_angular_margin_softmax(self, features, labels, is_training=None, reuse_variables=None, name="softmax"):
        """Additive angular margin softmax (ArcFace)
        link: https://arxiv.org/abs/1801.07698
        Annealing scheme is also added.
    
        Args:
            features: A tensor with shape [batch, dim].
            labels: A tensor with shape [batch].
            num_outputs: The number of classes.
            params: params.weight_l2_regularizer: the L2 regularization.
                    arcsoftmax_m: the angular margin (0.4-0.55)
                    params.arcsoftmax_norm, params.arcsoftmax_s: If arcsoftmax_norm is True, arcsoftmax_s must be specified.
                                                             This means we normalize the length of the features, and do the
                                                             scaling on the cosine similarity.
            is_training: Not used in this case.
            reuse_variables: Reuse variables.
            name:
        """
        #assert len(self.shape_list(features)) == len(self.shape_list(labels)) + 1
        num_outputs = self.num_classes
        # Convert the parameters to float
        arcsoftmax_lambda_min = float(0)
        arcsoftmax_lambda_base = float(1000)
        arcsoftmax_lambda_gamma = float(0.00001)
        arcsoftmax_lambda_power = float(5)
        arcsoftmax_m = float(0.00)
    
        tf.logging.info("Additive angular margin softmax is used.")
        tf.logging.info("The margin in the additive angular margin softmax is %f" % arcsoftmax_m)
    
        weight_l2_regularizer = 1e-2
        with tf.variable_scope(name, reuse=reuse_variables):
            w = tf.get_variable("output/kernel", [512, num_outputs], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
    
            w_norm = tf.nn.l2_normalize(w, dim=0)
            logits = tf.matmul(features, w_norm)
    
            ordinal = tf.to_int32(tf.range(self.batch_size))
            labels = tf.to_int32(tf.argmax(labels,1))
            ordinal_labels = tf.stack([ordinal, labels], axis=1)
            sel_logits = tf.gather_nd(logits, ordinal_labels)
    
            # The angle between x and the target w_i.
            eps = 1e-12
            features_norm = tf.maximum(tf.norm(features, axis=1), eps)
            cos_theta_i = tf.div(sel_logits, features_norm)
            cos_theta_i = tf.clip_by_value(cos_theta_i, -1+eps, 1-eps)  # for numerical steady
    
            # Since 0 < theta < pi, sin(theta) > 0. sin(theta) = sqrt(1 - cos(theta)^2)
            # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
            sin_theta_i_sq = 1 - tf.square(cos_theta_i)
            sin_theta_i = tf.sqrt(tf.maximum(sin_theta_i_sq, 1e-12))
            cos_theta_plus_m_i = cos_theta_i * tf.cos(arcsoftmax_m) - sin_theta_i * tf.sin(arcsoftmax_m)
    
            # Since theta \in [0, pi], theta + m > pi means cos(theta) < cos(pi - m)
            # If theta + m < pi, Phi(theta) = cos(theta + m).
            # If theta + m > pi, Phi(theta) = -cos(theta + m) - 2
            phi_i = tf.where(tf.greater(cos_theta_i, tf.cos(np.pi - arcsoftmax_m)),
                             cos_theta_plus_m_i,
                             -cos_theta_plus_m_i - 2)
    
            # logits = ||x||(cos(theta + m))
            scaled_logits = tf.multiply(phi_i, features_norm)
    
            logits_arcsoftmax = tf.add(logits,
                                       tf.scatter_nd(ordinal_labels,
                                                     tf.subtract(scaled_logits, sel_logits),
                                                     tf.shape(logits, out_type=tf.int32)))
    
            arcsoftmax_lambda = tf.maximum(arcsoftmax_lambda_min,
                                           arcsoftmax_lambda_base * (1.0 + arcsoftmax_lambda_gamma * tf.to_float(
                                               self.global_step)) ** (-arcsoftmax_lambda_power))
            fa = 1.0 / (1.0 + arcsoftmax_lambda)
            fs = 1.0 - fa
            updated_logits = fs * logits + fa * logits_arcsoftmax
    
            tf.summary.scalar("arcsoftmax_m", arcsoftmax_m)
            tf.summary.scalar("arcsoftmax_lambda", arcsoftmax_lambda)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=updated_logits)
            tf.summary.scalar("arcsoftmax_loss", loss)
    
        return logits, loss

    def softmax(self, features, labels, is_training=None, reuse_variables=None, name="softmax"):
        """Vanilla softmax loss.

        Args:
            features: A tensor with shape [batch, dim].
            labels: A tensor with shape [batch].
            num_outputs: The number of classes.
            params: params.weight_l2_regularizer used for L2 regularization.
            is_training: Not used in this case
            reuse_variables: Share the created variables or not.
            name:
        :return: A scalar tensor which is the loss value.
        """
        #assert len(shape_list(features)) == len(shape_list(labels)) + 1
        num_outputs = self.num_classes

        weight_l2_regularizer = 1e-2
        labels = tf.to_int32(tf.argmax(labels,1))
        with tf.variable_scope(name, reuse=reuse_variables):
            logits = tf.layers.dense(features,
                                       num_outputs,
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer),
                                       name="output")
            loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        return logits, loss

