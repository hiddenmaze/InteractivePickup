import tensorflow as tf
import numpy as np
from hourglass_with_rnn import createModel

def define_model(dim_sentence, max_step_sentence, num_hg_Depth, dim_hg_feat, dim_rnn_cell):
    img_resize = 256
    heatmap_resize = 64

    ph_image = tf.placeholder(dtype=tf.float32, shape=[None, img_resize, img_resize, 3])
    ph_sen = tf.placeholder(dtype=tf.float32, shape=[None, dim_sentence, max_step_sentence])
    ph_sen_len = tf.placeholder(tf.int32, [None, 1])
    ph_dropout = tf.placeholder(tf.float32)
    
    result_heatmap = createModel(curr_img = ph_image,
                                 curr_sen = ph_sen,
                                 curr_sen_len = ph_sen_len,
                                 img_size = heatmap_resize, 
                                 dim_sentence = dim_sentence,
                                 max_step_sentence = max_step_sentence, 
                                 num_hg_Depth = num_hg_Depth,
                                 dim_hg_feat = dim_hg_feat,
                                 dim_rnn_cell = dim_rnn_cell,
                                 dim_output = 1,
                                 dr_rate = ph_dropout
                                )
    return result_heatmap, ph_image, ph_sen, ph_sen_len, ph_dropout
    
def session_run(result_heatmap, ph_image, ph_sen, ph_sen_len, ph_dropout,
                restore_path, curr_embed_input, curr_seq_len, curr_test_img):
    init = tf.global_variables_initializer()
    
    HGN_vars = [v for v in tf.trainable_variables() if v.name.startswith('HGN')]    

    saver = tf.train.Saver(var_list = HGN_vars)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    T = 100

    with tf.Session(config=config) as sess:
        sess.run(init)

        saver.restore(sess, restore_path)

        test_feed_dict = {ph_sen: np.tile(curr_embed_input, (T, 1, 1)), 
                          ph_sen_len: np.tile(curr_seq_len, (T, 1)),
                          ph_image: np.tile(curr_test_img, (T, 1, 1, 1)), 
                          ph_dropout: 0.1}
        test_heatmap = sess.run(result_heatmap, feed_dict=test_feed_dict)

    test_heatmap = np.squeeze(test_heatmap)
    mean_of_esti = np.mean(test_heatmap, axis=0)
    mean_of_squared_esti = np.mean(test_heatmap**2, axis=0)
    squared_mean_of_esti = mean_of_esti ** 2
    uncertainty = np.sqrt(mean_of_squared_esti - squared_mean_of_esti)
    
    return mean_of_esti, uncertainty
