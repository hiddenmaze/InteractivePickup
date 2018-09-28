import numpy as np
import random
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import listdir, remove
from os.path import isfile, join
from google_drive_downloader import GoogleDriveDownloader as gdd
import gzip
import gensim

def load_w2v(w2v_path):
    if isfile(w2v_path) == False:
        print "Start downloading Google Word2Vec data"
        gdd.download_file_from_google_drive(file_id='0B7XkCwpI5KDYNlNUTTlSS21pQmM',
                                            dest_path=w2v_path+'.gz',
                                            unzip=False)
        inF = gzip.open(w2v_path+'.gz', 'rb')
        outF = open(w2v_path, 'wb')
        outF.write( inF.read() )
        inF.close()
        outF.close()

        remove(w2v_path+'.gz')
    print "Start loading Google Word2Vec data"
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    print "Finished loading Google Word2Vec data"

    return w2v_model

def QGN_organize_data(img_path):
    img_resize = 64
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]

    total_images = dict()
    for f in img_files:
        tmp_img = io.imread(join(img_path, f))
        tmp_img = resize(tmp_img, [img_resize, img_resize], preserve_range=True)
        tmp_img = tmp_img / 255.0
        total_images[f[0:4]] = tmp_img
    
    return total_images
    
def HGN_organize_data(img_path, script_path):
    img_resize = 256
    heatmap_resize = 64
    
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    script_files = [f for f in listdir(script_path) if isfile(join(script_path, f))]

    total_images = dict()
    for f in img_files:
        tmp_img = io.imread(('%s/%s') % (img_path, f))
        tmp_img = resize(tmp_img, [img_resize, img_resize], preserve_range=True)
        tmp_img = tmp_img / 255.0
        total_images[f[0:4]] = tmp_img

    total_heatmaps = dict()
    for f in script_files:
        npzfile = np.load(('%s/%s') % (script_path, f))
        tmp_heatmap = npzfile['arr_0']
        tmp_heatmap = resize(tmp_heatmap, [heatmap_resize, heatmap_resize], preserve_range=True)
        total_heatmaps[f[0:12]] = tmp_heatmap
    return total_images, total_heatmaps
  
def divide_img_idx(img_idx, num_data):
    if isfile('./data/divide_img_idx.npz') == False:
        random_seed=1
        divide_rate=20

        tmp_img_idx = range(max(img_idx)+1)
        random.seed(random_seed)
        random.shuffle(tmp_img_idx)

        num_test_img = int((max(img_idx)+1)/divide_rate)
        test_img_idx = tmp_img_idx[0:num_test_img]
        train_img_idx = tmp_img_idx[num_test_img:len(tmp_img_idx)]

        np.savez('./data/divide_img_idx.npz', train_img_idx, test_img_idx)
    else:
        npzfile = np.load('./data/divide_img_idx.npz')
        train_img_idx = npzfile['arr_0']
        test_img_idx = npzfile['arr_1']
                               
    print 'Test image index : %s' % (test_img_idx)

    return test_img_idx, train_img_idx
    
def get_test_idx(img_idx, num_data):
    test_img_idx, train_img_idx = divide_img_idx(img_idx, num_data)

    idx_whole = range(num_data)
    idx_test = []
    for i in test_img_idx:
        idx_test += np.where(img_idx == i)[0].tolist()
    idx_train = [i for i in idx_whole if i not in idx_test]
    
    return idx_test, idx_train

def QGN_divide_train_test(img_idx, sen_len, question_embeds, esti_maps, uncertainty_maps, answer_labels, num_data):
    idx_test, idx_train = get_test_idx(img_idx, num_data)

    num_train = len(idx_train)
    train_img_idx = img_idx[idx_train, :]
    train_sen_len = sen_len[idx_train, :]
    train_embeds = question_embeds[idx_train, :, :]
    train_esti = esti_maps[idx_train, :, :]
    train_uncertainty = uncertainty_maps[idx_train, :, :]
    train_answer_labels = answer_labels[idx_train, :]
    
    num_test = len(idx_test)
    test_img_idx = img_idx[idx_test, :]
    test_sen_len = sen_len[idx_test, :]
    test_embeds = question_embeds[idx_test, :, :]
    test_esti = esti_maps[idx_test, :, :]
    test_uncertainty = uncertainty_maps[idx_test, :, :]
    test_answer_labels = answer_labels[idx_test, :]

    print('Ended divided training and test dataset. Training : %d, Test : %d' % (num_train, num_test))                       
    
    return idx_train, idx_test, num_train, num_test,\
train_img_idx, train_sen_len, train_embeds,\
train_esti, train_uncertainty, train_answer_labels,\
test_img_idx, test_sen_len, test_embeds, test_esti, test_uncertainty, test_answer_labels
            
def HGN_divide_train_test(img_idx, sen_len, text_inputs, pos_outputs, num_data):
    idx_test, idx_train = get_test_idx(img_idx, num_data)

    num_train = len(idx_train)
    train_img_idx = img_idx[idx_train, :]
    train_sen_len = sen_len[idx_train, :]
    train_text_inputs = text_inputs[idx_train, :, :]
    train_pos_outputs = pos_outputs[idx_train, :]
    
    num_test = len(idx_test)    
    test_img_idx = img_idx[idx_test, :]
    test_sen_len = sen_len[idx_test, :]
    test_text_inputs = text_inputs[idx_test, :, :]
    test_pos_outputs = pos_outputs[idx_test, :]
    print('Ended divided training and test dataset. Training : %d, Test : %d' % (num_train, num_test))
    
    return idx_train, idx_test, num_train, num_test, \
           train_img_idx, train_sen_len, train_text_inputs, train_pos_outputs, \
           test_img_idx, test_sen_len, test_text_inputs, test_pos_outputs,
           

def load_test_img(img_path, curr_test_img_idx, img_resize, plot_flag):
    curr_test_img = np.zeros((1, img_resize, img_resize, 3))
    tmp_img = io.imread((img_path + '/%04d.jpg') % curr_test_img_idx)
    tmp_img = resize(tmp_img, [img_resize, img_resize], preserve_range=True)
    tmp_img = tmp_img / 255.0
    curr_test_img[0, :, :, :] = tmp_img

    if plot_flag == 1:
        plt.imshow(tmp_img)
        plt.show()
    
    return curr_test_img
    
def load_test_script(curr_test_input, w2v_model, dim_sentence, max_step_sentence):
    curr_embed_input = np.zeros((1, dim_sentence, max_step_sentence))
    curr_seq_len = np.zeros((1, 1))
    curr_words = curr_test_input.split()
    for i, word in enumerate(curr_words):
        if word not in w2v_model.vocab.keys():
            curr_embed_input[0, :, i] = np.zeros((300,))
        else:
            curr_embed_input[0, :, i] = w2v_model[word]
    curr_seq_len[0, 0] = len(curr_words)

    print "Ready the test input script"
    
    return curr_embed_input, curr_seq_len

def plot_HGN_result(curr_test_input, curr_test_img, mean_of_esti, uncertainty, bound_u=0.0, bound_c=0.0):
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    fig.suptitle('Input script: ' + curr_test_input, size=30)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(resize(curr_test_img[0, :, :, :], [256, 256], preserve_range=True))
    ax1.set_title('input image', size=20)
    ax1.set_axis_off()

    ax2 = fig.add_subplot(2, 2, 2)
    plot2 = ax2.imshow(resize(mean_of_esti, [256, 256], preserve_range=True), cmap='jet')
    ax2.set_title('estimation map', size=20)
    ax2.set_axis_off()
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plot2, ax=ax2, cax=cax2)

    ax3 = fig.add_subplot(2, 2, 3)
    if bound_u > 0:
        norm3 = mpl.colors.Normalize(vmin=0, vmax=bound_u)
        plot3 = ax3.imshow(resize(uncertainty, [256, 256], preserve_range=True), cmap='jet',
                          norm = norm3)
    else:
        norm3 = mpl.colors.Normalize(vmin=0, vmax=np.max(uncertainty))
        plot3 = ax3.imshow(resize(uncertainty, [256, 256], preserve_range=True), cmap='jet',
                          norm = norm3)
    ax3.set_title('uncertainty map', size=20)
    ax3.set_axis_off()
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plot3, ax=ax3, cax=cax3)   
    
    ax4 = fig.add_subplot(2, 2, 4)
    if bound_c > 0:
        norm4 = mpl.colors.Normalize(vmin=0, vmax=bound_c)
        plot4 = ax4.imshow(resize(2*uncertainty + mean_of_esti, [256, 256], preserve_range=True), cmap='jet', norm=norm4)
    else:
        norm4 = mpl.colors.Normalize(vmin=0, vmax=np.max(2*uncertainty + mean_of_esti))
        plot4 = ax4.imshow(resize(2*uncertainty + mean_of_esti, [256, 256], preserve_range=True), cmap='jet', norm=norm4) 
    ax4.set_title('confidence map', size=20)
    ax4.set_axis_off()
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plot4, ax=ax4, cax=cax4)
 