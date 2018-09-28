import tensorflow as tf
import numpy as np
from vgg_with_rnn import fused_model, rnn_encoder

def define_model(dim_sentence, max_step_sentence, dim_rnn_cell, num_label):
    img_resize = 64
    
    ph_combine = tf.placeholder(dtype=tf.float32, shape=[None, img_resize, img_resize, 4])
    ph_sen = tf.placeholder(dtype=tf.float32, shape=[None, dim_sentence, max_step_sentence])
    ph_sen_len = tf.placeholder(tf.int32, [None, 1])
    ph_keepprob = tf.placeholder(tf.float32)

    my_rnn_encoder = rnn_encoder(dim_rnn_cell = dim_rnn_cell,
                                 dim_sentence = dim_sentence,
                                 max_step_sentence = max_step_sentence)
    rnn_feature = my_rnn_encoder.encode(ph_sen, ph_sen_len)

    my_fused_model = fused_model()

    question_logits = my_fused_model.vgg_with_rnn(rnn_feature = rnn_feature,
                                               init_image = ph_combine,
                                               keep_prob = ph_keepprob,
                                               n_classes = num_label)
    
    return question_logits, ph_combine, ph_sen, ph_sen_len, ph_keepprob

def session_run(question_logits, ph_combine, ph_sen, ph_sen_len, ph_keepprob,
               restore_path, curr_combine_map, curr_embed_input, curr_seq_len):
    init = tf.global_variables_initializer()

    QGN_vars = [v for v in tf.trainable_variables() if v.name.startswith('QGN')]    

    saver = tf.train.Saver(var_list=QGN_vars)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        saver.restore(sess, restore_path)

        test_feed_dict={ph_combine: curr_combine_map, ph_sen: curr_embed_input,
                        ph_sen_len: curr_seq_len, ph_keepprob: 1.0}
        curr_question = sess.run(tf.nn.softmax(question_logits), feed_dict=test_feed_dict)
        
    return curr_question
