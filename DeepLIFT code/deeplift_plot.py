import deeplift
import re
import numpy as np
from collections import OrderedDict
from deeplift.visualization import viz_sequence
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.layers import NonlinearMxtsMode
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle 

def One_hot(dirs):
    files = open(dirs, 'r')
    sample = []
    for line in files:
        if re.match('>', line) is None:
            value = np.zeros((170, 4), dtype='float32')
            for index, base in enumerate(line.strip()):
                if re.match(base, 'A|a'):
                    value[index, 0] = 1
                if re.match(base, 'C|c'):
                    value[index, 1] = 1
                if re.match(base, 'G|g'):
                    value[index, 2] = 1
                if re.match(base, 'T|t'):
                    value[index, 3] = 1
            sample.append(value)
    files.close()
    return np.array(sample)

def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,sequence=sequence, one_hot_axis=1)

    return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1): 
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1

if __name__ == "__main__":
    model_path = './cnn_densenet_leaf1.h5'
    index_path = './selected_index.txt'
    file_path = './selected_fasta.fa'
    X = One_hot(file_path)
    with open(file_path) as f:
        data = []
        while True:
            line = f.readline()
            if not line:
                break
            if not line[0] == '>':
                data.append(line.strip())

    deeplift_model = kc.convert_model_from_saved_files(model_path,
                    nonlinear_mxts_mode = deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                    find_scores_layer_name = 'input_2_0',
                    pre_activation_target_layer_name = 'dense_1_0')

    rescale_conv_revealcancel_fc_many_refs_func = get_shuffle_seq_ref_function(
                    score_computation_function = deeplift_contribs_func,
                    shuffle_func=dinuc_shuffle,
                    one_hot_func=lambda x: np.array([one_hot_encode_along_channel_axis(seq) for seq in x]))

    num_refs_per_seq=20 
    scores_without_sum_applied = rescale_conv_revealcancel_fc_many_refs_func(
                    task_idx=0, 
                    input_data_sequences=data,
                    num_refs_per_seq=num_refs_per_seq,
                    batch_size=200,
                    progress_update=1000)

    scores = np.sum(scores_without_sum_applied,axis=2)

    index = np.loadtxt(index_path).astype(np.int32)

    for idx in range(len(data)):
        print("Scores for example:{}\n",index[idx])
        scores_for_idx = scores[idx]
        original_onehot = X[idx]
        scores_for_idx = original_onehot*scores_for_idx[:,None]
        viz_sequence.plot_weights(scores_for_idx, subticks_frequency=10,figsize=(100,10))
                                  #save_fig = True,save_dir='D:/leaf'+str(index[idx])+'.eps')#, highlight=highlight)


