import tensorflow as tf
import numpy as np
from vgg_with_rnn import fused_model, rnn_encoder
import random
import time

def train(img_resize, dim_sentence, max_step_sentence, 
         batch_size, max_epoch, num_train, save_stride,
         dim_rnn_cell, num_label, learning_rate,
         restore_flag, restore_path, restore_epoch,
         total_images, train_embeds, train_esti, train_uncertainty, train_question_labels,
         train_sen_len, train_img_idx):    
    
    num_batch = num_train / batch_size
    
    ph_combine = tf.placeholder(dtype=tf.float32, shape=[None, img_resize, img_resize, 4])
    ph_sen = tf.placeholder(dtype=tf.float32, shape=[None, dim_sentence, max_step_sentence])
    ph_sen_len = tf.placeholder(tf.int32, [None, 1])
    ph_question_label = tf.placeholder(dtype=tf.int32, shape=[None, 1])
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

    loss = tf.losses.sparse_softmax_cross_entropy(logits=question_logits, 
                                                  labels=ph_question_label)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    print 'Now ready to start the session.'

    with tf.Session(config=config) as sess:
        sess.run(init)
        if restore_flag == 1:
            saver.restore(sess, restore_path)
        for _epoch in xrange(max_epoch-restore_epoch):
            random.seed(_epoch)
            batch_shuffle = range(num_train)
            random.shuffle(batch_shuffle)

            total_train_loss = 0.0

            epoch_start_time = time.time()
            for i in xrange(num_batch):
                batch_idx = [batch_shuffle[idx] 
                             for idx in range(i * batch_size, (i + 1) * batch_size)]
                batch_embed = train_embeds[batch_idx, :, :]
                batch_esti = train_esti[batch_idx, :, :]
                batch_uncertainty = train_uncertainty[batch_idx, :, :]
                batch_question_labels = train_question_labels[batch_idx, :]
                batch_sen_len = train_sen_len[batch_idx, :]
                batch_img_idx = train_img_idx[batch_idx, :]

                batch_combine_map = np.zeros((batch_size, img_resize, img_resize, 4))

                for ii in xrange(len(batch_idx)):
                    tmp_img = total_images['%04d' % batch_img_idx[ii, 0]]

                    batch_upperbound = batch_esti[ii, :, :] + 2 * (batch_uncertainty[ii, :, :])
                    batch_combine_map[ii, :, :, 0:3] = tmp_img
                    batch_combine_map[ii, :, :, 3] = batch_upperbound

                train_feed_dict={ph_combine: batch_combine_map,
                                ph_sen: batch_embed,
                                ph_sen_len: batch_sen_len,
                                ph_question_label: batch_question_labels,
                                ph_keepprob: 0.8}
                sess.run(optimizer, feed_dict=train_feed_dict)
                curr_train_loss = sess.run(loss, feed_dict=train_feed_dict)
                total_train_loss += curr_train_loss

                batch_end_time = time.time()
                total_time = batch_end_time - epoch_start_time
                if i % 500 == 0:
                    print("batch loss : %s -> about %0.3f second left to finish this epoch" 
                          % (curr_train_loss, (total_time/(i+1))*(num_batch-i) ))


            total_train_loss = total_train_loss / num_batch

            print('current epoch : ' + str(_epoch+1+restore_epoch), 
                  ', current train loss : ' + str(total_train_loss)) 
            if (_epoch+1+restore_epoch) % save_stride == 0:
                model_save_path = saver.save(sess, './trained_QGN/model.ckpt', 
                                             global_step=_epoch+1+restore_epoch)
                print("Model saved in file : %s" % model_save_path)