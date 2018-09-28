### Load Dependencies
import numpy as np
from os import listdir, remove
from os.path import isfile, join
from utils import load_w2v

def preprocess():        
    ### Load word2vec model
    w2v_path = './data/GoogleNews-vectors-negative300.bin'
    w2v_model = load_w2v(w2v_path)

    ### Define the path and file lists
    certain_question_path = './data/question_certain/'
    certain_data_path = './data/data_certain/'

    amb_question_path = './data/question_ambiguous/'
    amb_data_path ='./data/data_ambiguous/'

    cq_files = [f for f in listdir(certain_question_path) if isfile(join(certain_question_path, f))]
    cd_files = [f for f in listdir(certain_data_path) if isfile(join(certain_data_path, f))]

    aq_files = [f for f in listdir(amb_question_path) if isfile(join(amb_question_path, f))]
    ad_files = [f for f in listdir(amb_data_path) if isfile(join(amb_data_path, f))]

    ### Calculate maximum language command length and organize answer set
    max_len = 0
    num_data = 0
    question_set = set()

    for f in cq_files:
        npzfile = np.load(join(certain_question_path, f))
        curr_cq = npzfile['arr_0']
        curr_cq = curr_cq.item()

        curr_keys = curr_cq.keys()
        for key in curr_keys:
            words = key.split()
            if len(words) > max_len:
                max_len = len(words)
            curr_values = curr_cq[key]
            num_data += len(curr_values)
            for value in curr_values:
                question_set.add(value)

    for f in aq_files:
        npzfile = np.load(join(amb_question_path, f))
        curr_aq = npzfile['arr_0']
        curr_aq = curr_aq.item()

        curr_keys = curr_aq.keys()
        for key in curr_keys:
            words = key.split()
            if len(words) > max_len:
                max_len = len(words)
            curr_values = curr_aq[key]
            num_data += len(curr_values)
            for value in curr_values:
                question_set.add(value)
    max_len += 6
    print('Total data number: %d' % num_data)

    ### Organize answer labels
    question_label = {'this': 0,
                      'red': 1, 'yellow': 2, 'yelow': 2, 'green': 3, 'blue': 4, 'purple': 5,
                      'upper': 6, 'upperright': 7, 'right': 8, 'lowerright': 9,
                      'lower' : 10, 'lowerleft': 11, 'left': 12, 'upperleft': 13, 'middle': 14}
    reverse_label=['this', 'red', 'yellow', 'green', 'blue', 'purple',
                  'upper', 'upper right', 'right','lower right',
                  'lower', 'lower left', 'left', 'upper left','middle']
    num_question_cand = 15

    ### Define empty data arrays
    heatmap_size = 64
    dim_embed = w2v_model['woman'].shape[0]
    question_embeds = np.zeros((num_data, dim_embed, max_len)) 
    answer_labels = np.zeros((num_data, 1)) 
    esti_maps = np.zeros((num_data, heatmap_size, heatmap_size))
    uncertainty_maps = np.zeros((num_data, heatmap_size, heatmap_size))
    img_idx = np.zeros((num_data, 1)) 
    seq_len = np.zeros((num_data, 1)) 

    ### Start proprocessing
    tmp_num = 0
    print 'Start proprocessing for the question generation network'
    for f in cq_files:
        npzfile = np.load(join(certain_question_path, f))
        curr_cq = npzfile['arr_0']
        curr_cq = curr_cq.item()

        npzfile = np.load(join(certain_data_path, f))
        curr_cd = npzfile['arr_0']
        curr_cd = curr_cd.item()

        curr_keys = curr_cd.keys()

        for key in curr_keys:
            words = key.split()
            curr_values = curr_cq[key]
            for value in curr_values:
                for i, word in enumerate(words):
                    if word not in w2v_model.vocab.keys():
                        question_embeds[tmp_num, :, i] = np.zeros((300,))
                    else:
                        question_embeds[tmp_num, :, i] = w2v_model[word]
                img_idx[tmp_num, 0] = float(f[0:4])
                seq_len[tmp_num, 0] = len(words)
                esti_maps[tmp_num, :, :] = curr_cd[key]['mean']
                uncertainty_maps[tmp_num, :, :] = curr_cd[key]['uncertainty']
                question_labels[tmp_num, :] = question_label[value]
                tmp_num += 1

    for f in aq_files:
        npzfile = np.load(join(amb_question_path, f))
        curr_aq = npzfile['arr_0']
        curr_aq = curr_aq.item()

        npzfile = np.load(join(amb_data_path, f))
        curr_ad = npzfile['arr_0']
        curr_ad = curr_ad.item()

        curr_keys = curr_ad.keys()

        for key in curr_keys:
            words = key.split()
            curr_values = curr_aq[key]
            for value in curr_values:
                for i, word in enumerate(words):
                    if word == 'to' or word == 'and':
                        question_embeds[tmp_num, :, i] = np.zeros((300,))
                    else:
                        question_embeds[tmp_num, :, i] = w2v_model[word]
                img_idx[tmp_num, 0] = float(f[0:4])
                seq_len[tmp_num, 0] = len(words)
                esti_maps[tmp_num, :, :] = curr_ad[key]['mean']
                uncertainty_maps[tmp_num, :, :] = curr_ad[key]['uncertainty']
                question_labels[tmp_num, :] = question_label[value]
                tmp_num += 1    

    ### Save the proprocessed data
    np.savez('./data/preprocessed4QGN.npz',
             img_idx, seq_len, question_embeds, esti_maps, 
             uncertainty_maps, answer_labels, reverse_label
            )
    print 'Finished saving the preprocessed file for the question generation network'