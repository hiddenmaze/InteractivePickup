import tensorflow as tf

def convBlock(curr_input, numIn, numOut, dr_rate):
    net = tf.layers.batch_normalization(curr_input)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, numOut/2, 1, padding='Same')
    net = tf.layers.dropout(net, rate=dr_rate)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, numOut/2, 3, padding='Same')
    net = tf.layers.dropout(net, rate=dr_rate)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, numOut, 1, padding='Same')
    net = tf.layers.dropout(net, rate=dr_rate)

    return net

def skipLayer(curr_input, numIn, numOut, dr_rate):
    if numIn == numOut:
        return curr_input
    else:
        net = tf.layers.conv2d(curr_input, numOut, 1, padding='Same')
        net = tf.layers.dropout(net, rate=dr_rate)
        return net

def Residual(curr_input, numIn, numOut, dr_rate):
    curr_convBlock = convBlock(curr_input, numIn, numOut, dr_rate)
    curr_skipLayer = skipLayer(curr_input, numIn, numOut, dr_rate)
    return curr_convBlock + curr_skipLayer