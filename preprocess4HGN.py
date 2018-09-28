### Import Dependencies
import numpy as np
from os import listdir, remove
from os.path import isfile, join
from scipy import misc as misc
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mulnorm
from utils import load_w2v

### Import word2vec model
def preprocess():
    w2v_path = './data/GoogleNews-vectors-negative300.bin'
    w2v_model = load_w2v(w2v_path)

    dim_embed = w2v_model['woman'].shape[0]

    ### Calculate Maximum length of the language command
    script_path = './data/train_script'
    files = [f for f in listdir(script_path) if isfile(join(script_path, f))]

    total_words = set()
    max_len = 0
    num_data = 0

    for tmp_file in files:
        curr_file = open(join(script_path, tmp_file), 'r')
        curr_lines = curr_file.readlines()
        for line in curr_lines:
            words = line.split()
            for word in words:
                if words.index(word) > 1:
                    total_words.add(word)
            if max_len < len(words)-2:
                max_len = len(words)-2
        num_data += len(curr_lines)

    total_words = list(total_words)

    max_len = max_len + 5

    print('Total data number: %d' % num_data)

    ### Set Empty data arrays
    inputs = np.zeros((num_data, dim_embed, max_len))
    outputs = np.zeros((num_data, 2))
    img_idx = np.zeros((num_data, 1))
    seq_len = np.zeros((num_data, 1))
    tmp_num = 0

    ### Start Preprocess : delete # for re-generate heatmaps
    for file_idx in (files):
        curr_file = open(join(script_path, file_idx), 'r')
        curr_lines = curr_file.readlines()

        print 'Now processing : %s/%s ... ' % (files.index(file_idx), len(files))

        prev_output = [0, 0]
        for line in curr_lines:
            words = line.split()
            for i, word in enumerate(words):
                if i <= 1:
                    outputs[tmp_num, i] = float(word)
                else:
                    if word not in w2v_model.vocab.keys():
                        inputs[tmp_num, :, i-2] = np.zeros((300,))
                    else:
                        inputs[tmp_num, :, i-2] = w2v_model[word]
            img_idx[tmp_num, 0] = float(file_idx[0:4])
            seq_len[tmp_num, 0] = len(words)-2

            '''
            if prev_output[0] != outputs[tmp_num, 0] or prev_output[1] != outputs[tmp_num, 1]:
                tmp_heatmap = np.zeros((250, 250))
                tmp_output = outputs[tmp_num, :].astype(int)
                print 'generate heatmap at %03d, %03d' % (tmp_output[0], tmp_output[1])

                tmp_cov = [[500, 0], [0, 500]]
                mvn = mulnorm([tmp_output[1], tmp_output[0]], tmp_cov)

                for k in range(tmp_heatmap.shape[0]):
                    for kk in range(tmp_heatmap.shape[1]):
                        tmp_heatmap[k, kk] = mvn.pdf([k, kk])

                tmp_heatmap = tmp_heatmap / np.max(tmp_heatmap)
                np.savez(('./data/train_heatmap/%s_%03d_%03d.npz')%(file_idx[0:4], tmp_output[0], tmp_output[1] ), tmp_heatmap)

            prev_output = outputs[tmp_num, :]
            '''        
            tmp_num += 1        

    ### Save preprocessed data
    np.savez('./data/preprocessed4HGN.npz', img_idx, seq_len, inputs, outputs, total_words)
