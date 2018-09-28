import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs

stddev = 0.01

class fused_model(object):
    def __init__(self):
        print('Fused Model Created')
        
    def vgg_with_rnn(self, rnn_feature, init_image, keep_prob, n_classes):         
        print '=======VGG NET with RNN======='
        with vs.variable_scope("QGN/cnn"):
            self.conv_1_1 = tf.layers.conv2d(init_image, filters=16, kernel_size=3, padding='same')
            self.pool_1 = tf.layers.max_pooling2d(tf.nn.relu(self.conv_1_1), pool_size=2, strides=2)

            # print 'pool_1 -- shape : %s' % self.pool_1.shape

            # block 2 -- outputs 56 56 128
            self.conv_2_1 = tf.layers.conv2d(self.pool_1, filters=32, kernel_size=3, padding='same')
            self.pool_2 = tf.layers.max_pooling2d(tf.nn.relu(self.conv_2_1), pool_size=2, strides=2)

            # print 'pool_2 -- shape : %s' % self.pool_2.shape

            # block 3 -- outputs 28 28 256
            self.conv_3_1 = tf.layers.conv2d(self.pool_2, filters=64, kernel_size=3, padding='same')
            self.pool_3 = tf.layers.max_pooling2d(tf.nn.relu(self.conv_3_1), pool_size=2, strides=2)

            # print 'pool_3 -- shape : %s' % self.pool_3.shape

            # flatten
            flattened_shape = np.prod([s.value for s in self.pool_3.get_shape()[1:]])
            self.flatten = tf.reshape(self.pool_3, [-1, flattened_shape])
            # print 'flatten -- shape : %s' % self.flatten.shape

            # fully connected
            self.fc_1 = tf.layers.dense(self.flatten, units= 1024, activation=tf.nn.relu)
            # print 'fc_1 -- shape : %s' % self.fc_1.shape

            self.fuse_1 = tf.concat([self.fc_1, rnn_feature], axis=1)
            # print 'fuse_1 -- shape : %s' % self.fuse_1.shape

            self.drop_1 = tf.nn.dropout(self.fuse_1, keep_prob)

            self.fc_2 = tf.layers.dense(self.fuse_1, units = rnn_feature.shape[1],
                                        activation=tf.nn.relu)
            # print 'fc_2 -- shape : %s' % self.fc_2.shape

            self.fuse_2 = tf.multiply(self.fc_2, rnn_feature)
            # print 'fuse_2 -- shape : %s' % self.fuse_2.shape

            self.drop_2 = tf.nn.dropout(self.fuse_2, keep_prob)

            self.fc_3 = tf.layers.dense(self.drop_2, units = n_classes)
            # print 'fc_3 -- shape : %s' % self.fc_3.shape
        
        return self.fc_3
         
class rnn_encoder(object):
    def __init__(self, dim_rnn_cell, dim_sentence, max_step_sentence):
        self.dim_sentence = dim_sentence
        self.max_step_sentence = max_step_sentence
        self.dim_rnn_cell = dim_rnn_cell        
                    
        print('RNN ENCODER CREATED')                      

    def encode(self, _x, _sen_len):
        _x_split = tf.transpose(_x, [2, 0, 1])
        _x_split = tf.reshape(_x_split, [-1, self.dim_sentence])
        _x_split = tf.split(_x_split, self.max_step_sentence, axis=0)
        
        with vs.variable_scope("QGN/rnn"):
            _rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_rnn_cell)
            _output, _state = tf.nn.static_rnn(_rnn_cell, _x_split, dtype=tf.float32, sequence_length=tf.squeeze(_sen_len))

        return _state[-1]            
