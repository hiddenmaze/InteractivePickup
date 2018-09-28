import tensorflow as tf
from residual import Residual
from tensorflow.python.ops import variable_scope as vs

stddev = 0.01

def hourglass_with_rnn(curr_input, rnn_feat, numDepth, numIn, numOut, dr_rate):
    up1 = curr_input
    up1 = Residual(up1, numIn, numOut, dr_rate)

    low1 = tf.layers.max_pooling2d(curr_input, 2, 2)
    low1 = Residual(low1, numIn, numOut, dr_rate)
    
    if numDepth > 1:
        low2 = hourglass_with_rnn(low1, rnn_feat, numDepth-1, numIn, numOut, dr_rate)
    else:
        low2 = Residual(low1, numIn, numOut, dr_rate)
    
    low3 = Residual(low2, numIn, numOut-1, dr_rate)
    
    with tf.variable_scope("word_weights", reuse=True):
        exec('_W_o%d = tf.get_variable('"'W_o%d'"')' % (numDepth, numDepth))
        exec('_b_o%d = tf.get_variable('"'b_o%d'"')' % (numDepth, numDepth))
    
    exec('fitted_rnn = tf.matmul(rnn_feat, _W_o%d)+_b_o%d' % (numDepth, numDepth))  
    fitted_rnn = tf.nn.dropout(fitted_rnn, 1-dr_rate)
    
    fitted_rnn = tf.reshape(fitted_rnn, [tf.shape(rnn_feat)[0], 
                                        tf.to_int32(tf.sqrt(tf.to_float(tf.shape(fitted_rnn)[1]))),
                                        tf.to_int32(tf.sqrt(tf.to_float(tf.shape(fitted_rnn)[1])))])
    fitted_rnn = tf.expand_dims(fitted_rnn, axis=3)
    
    fuse_low3 = tf.concat([low3, fitted_rnn], axis=3)
        
    up2 = tf.image.resize_nearest_neighbor(fuse_low3, 2*tf.shape(fuse_low3)[1:3])

    return tf.add(up1, up2)


def lin(curr_input, numIn, numOut, dr_rate):
    l = tf.layers.conv2d(curr_input, numOut, 1, padding='Same')
    l = tf.layers.dropout(l, rate=dr_rate)
    l = tf.layers.batch_normalization(l)
    return tf.nn.relu(l)


def createModel(curr_img, curr_sen, curr_sen_len,
                img_size, dim_sentence, max_step_sentence, 
                num_hg_Depth, dim_hg_feat, dim_rnn_cell, dim_output,
                dr_rate
                ):
    # image size must be 256 by 256.
    curr_rnn_encoder= rnn_encoder(img_size, num_hg_Depth, dim_sentence, max_step_sentence, dim_rnn_cell, dr_rate)
    
    rnn_feat = curr_rnn_encoder.encode(curr_sen, curr_sen_len)
    
    with vs.variable_scope('HGN'):
        with vs.variable_scope('pre'):
            cnv1 = tf.layers.conv2d(curr_img, filters=dim_hg_feat/4, kernel_size=7, strides=2, padding='Same')
            cnv1 = tf.layers.dropout(cnv1, rate=dr_rate)

            cnv1 = tf.layers.batch_normalization(cnv1)
            cnv1 = tf.nn.relu(cnv1)
        
        with vs.variable_scope('r1'):
            r1 = Residual(cnv1, dim_hg_feat/4, dim_hg_feat/2, dr_rate)

        pool = tf.layers.max_pooling2d(r1, 2, 2)

        with vs.variable_scope('r4'):    
            r4 = Residual(pool, dim_hg_feat/2, dim_hg_feat/2, dr_rate)

        with vs.variable_scope('r5'):    
            r5 = Residual(r4, dim_hg_feat/2, dim_hg_feat, dr_rate)    
    
        with vs.variable_scope('hg'):  
            hg = hourglass_with_rnn(r5, rnn_feat, num_hg_Depth, dim_hg_feat, dim_hg_feat, dr_rate)

        with vs.variable_scope('ll'):      
            ll = Residual(hg, dim_hg_feat, dim_hg_feat, dr_rate)
            ll = lin(ll, dim_hg_feat, dim_hg_feat, dr_rate)
            
        with vs.variable_scope('out'): 
            Out = tf.layers.conv2d(ll, filters=dim_output, kernel_size=1, strides=1, padding='Same')
            Out = tf.layers.dropout(Out, rate=dr_rate)
    
    return Out


class rnn_encoder(object):
    def __init__(self, img_size, hg_depth, dim_sentence, max_step_sentence, dim_rnn_cell, dr_rate):
        self.img_size = img_size
        self.hg_depth = hg_depth
        self.dim_sentence = dim_sentence
        self.max_step_sentence = max_step_sentence
        self.dim_rnn_cell = dim_rnn_cell
        self.dr_rate = dr_rate
        
        with tf.variable_scope("HGN/hg/word_weights"):                                              
            for i in xrange(hg_depth):
                exec('self.W_o%d = tf.get_variable('"'W_o%d'"', dtype=tf.float32,\
                                        initializer=tf.random_normal([self.dim_rnn_cell, \
                                                    (self.img_size/(2**(self.hg_depth+1-%d)))**2], \
                                                    stddev=stddev))' % (i+1, i+1, i+1))
                exec('self.b_o%d = tf.get_variable('"'b_o%d'"', dtype=tf.float32,\
                                       initializer=tf.random_normal([(self.img_size/(2**(self.hg_depth+1-%d)))**2], \
                                                                     stddev=stddev))' 
                                                                     % (i+1, i+1, i+1))
    def encode(self, _x, _sen_len):
        _x_split = tf.transpose(_x, [2, 0, 1])
        _x_split = tf.reshape(_x_split, [-1, self.dim_sentence])
        _x_split = tf.split(_x_split, self.max_step_sentence, axis=0)

        with vs.variable_scope("HGN/rnn"):
            _rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_rnn_cell)
            _rnn_cell = tf.nn.rnn_cell.DropoutWrapper(_rnn_cell, output_keep_prob=1-self.dr_rate, input_keep_prob=1-self.dr_rate, state_keep_prob=1-self.dr_rate)
            _output, _state = tf.nn.static_rnn(_rnn_cell, _x_split, dtype=tf.float32, sequence_length=tf.squeeze(_sen_len))

        return _state[-1]     