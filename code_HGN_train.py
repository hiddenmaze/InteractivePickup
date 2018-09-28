import tensorflow as tf
import numpy as np
import random
import time
from hourglass_with_rnn import createModel

def train(img_resize, heatmap_resize, dim_sentence, max_step_sentence,
          batch_size, max_epoch, num_train, save_stride,
          num_hg_Depth, dim_hg_feat, dim_rnn_cell,
          restore_flag, restore_path, restore_epoch,
          total_images, total_heatmaps,
          train_text_inputs, train_sen_len, train_img_idx, train_pos_outputs,
          learning_rate):
    
    num_batch = num_train / batch_size
    
    ph_image = tf.placeholder(dtype=tf.float32, shape=[None, img_resize, img_resize, 3])
    ph_sen = tf.placeholder(dtype=tf.float32, shape=[None, dim_sentence, max_step_sentence])
    ph_sen_len = tf.placeholder(tf.int32, [None, 1])
    ph_heatmap = tf.placeholder(dtype=tf.float32, shape=[None, heatmap_resize, heatmap_resize, 1])
    ph_dropout = tf.placeholder(tf.float32)

    result_heatmap = createModel(curr_img = ph_image,
                                 curr_sen = ph_sen,
                                 curr_sen_len = ph_sen_len,
                                 img_size=heatmap_resize, 
                                 dim_sentence=dim_sentence,
                                 max_step_sentence=max_step_sentence, 
                                 num_hg_Depth=num_hg_Depth,
                                 dim_hg_feat=dim_hg_feat,
                                 dim_rnn_cell=dim_rnn_cell,
                                 dim_output=1,
                                 dr_rate=ph_dropout,
                                )
    loss = tf.reduce_mean((result_heatmap-ph_heatmap)**2)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

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
                batch_inputs = train_text_inputs[batch_idx, :, :]
                batch_seq_len = train_sen_len[batch_idx, :]
                batch_img_idx = train_img_idx[batch_idx, :]
                batch_pos_output = train_pos_outputs[batch_idx, :]

                batch_images = np.zeros((batch_size, img_resize, img_resize, 3))
                batch_heatmaps = np.zeros((batch_size, heatmap_resize, heatmap_resize, 1))

                for ii in xrange(len(batch_idx)):
                    tmp_img = total_images['%04d' % batch_img_idx[ii, 0]]

                    batch_images[ii, :, :, :] = tmp_img        

                    tmp_heatmap = total_heatmaps['%04d_%03d_%03d' 
                                                 % (batch_img_idx[ii, 0], 
                                                    batch_pos_output[ii, 0], 
                                                    batch_pos_output[ii, 1])]  

                    batch_heatmaps[ii, :, :, 0] = tmp_heatmap

                train_feed_dict = {ph_sen: batch_inputs, ph_sen_len: batch_seq_len,
                                   ph_image: batch_images, ph_heatmap: batch_heatmaps,
                                   ph_dropout: 0.0}

                sess.run(optimizer, feed_dict=train_feed_dict)
                curr_train_loss = sess.run(loss, feed_dict=train_feed_dict)
                total_train_loss += curr_train_loss

                batch_end_time = time.time()
                total_time = batch_end_time - epoch_start_time 
                if i % 100 == 0:
                    print("batch loss : %s -> about %0.3f second left to finish this epoch" 
                          % (curr_train_loss, (total_time/(i+1))*(num_batch-i) ))

            total_train_loss = total_train_loss / num_batch

            print('current epoch : ' + str(_epoch+1+restore_epoch), 
                  ', current train loss : ' + str(total_train_loss)) 
            if (_epoch+1+restore_epoch) % save_stride == 0:
                model_save_path = saver.save(sess, './trained_HGN/model.ckpt', 
                                             global_step=_epoch+1+restore_epoch)
                print("Model saved in file : %s" % model_save_path)
