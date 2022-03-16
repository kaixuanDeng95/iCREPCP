
import numpy as np
import pandas as pd
df_leaf_train = pd.read_csv('CNN_train_leaf.orig.tsv',sep='\t')
df_leaf_test = pd.read_csv('CNN_test_leaf.orig.tsv',sep='\t')
df = pd.concat([df_leaf_train,df_leaf_test], axis=0)
to_save = df.iloc[:,2:4]
to_save.to_csv('leaf.csv',header=False,index=False,sep='\t')

# One-hot encoding
import numpy as np

#this is set up for 1d convolutions where examples
#have dimensions (len, num_channels)
#the channel axis is the axis for one-hot encoding.
def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
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
            

fasta_sequences = []
for i,a_line in enumerate(open("leaf.csv","r")):
    a_line = a_line.rstrip()
    seq_id,seq_fasta = a_line.split("\t")
    fasta_sequences.append(seq_fasta)

onehot_data = np.array([one_hot_encode_along_channel_axis(seq)
                        for seq in fasta_sequences])

print(onehot_data.shape)

# Prepare the deeplift models
import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.layers import NonlinearMxtsMode
from tensorflow.keras.models import load_model

deeplift_model =kc.convert_model_from_saved_files(
        'cnn_densenet_leaf1.h5',
        nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

# Prepare the deeplift models
import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.layers import NonlinearMxtsMode

deeplift_model =kc.convert_model_from_saved_files(
        'cnn_densenet_leaf1.h5',
        nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

# Compute importance scores
# Compile the DeepLIFT contribution scoring function
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle #function to do a dinucleotide shuffle
contribs_func = deeplift_model.get_target_contribs_func(
    find_scores_layer_name='input_2_0',
    pre_activation_target_layer_name='dense_1_0')

# Adapt the scoring function to work with multiple shuffled references
contribs_many_refs_func = get_shuffle_seq_ref_function(
    #score_computation_function is the original function to compute scores
    score_computation_function=contribs_func,
    #shuffle_func is the function that shuffles the sequence
    #On real genomic data, a dinuc shuffle is advisable due to
    #the strong bias against CG dinucleotides
    shuffle_func=dinuc_shuffle,
    one_hot_func=lambda x: np.array([one_hot_encode_along_channel_axis(seq)
                                     for seq in x]))

# Compile the "hypothetical" contribution scoring function
from deeplift.util import get_hypothetical_contribs_func_onehot

multipliers_func = deeplift_model.get_target_multipliers_func(find_scores_layer_name='input_2_0',
                                                              pre_activation_target_layer_name='dense_1_0')
hypothetical_contribs_func = get_hypothetical_contribs_func_onehot(multipliers_func)

#Once again, we rely on multiple shuffled references
hypothetical_contribs_many_refs_func = get_shuffle_seq_ref_function(
    score_computation_function=hypothetical_contribs_func,
    shuffle_func=dinuc_shuffle,
    one_hot_func=lambda x: np.array([one_hot_encode_along_channel_axis(seq)
                                     for seq in x]))

# Obtain the scores
num_refs_per_seq = 10
task_to_contrib_scores = {}
task_to_hyp_contrib_scores = {}
all_tasks = [0]
for task_idx in all_tasks:
    print("On task",task_idx)
    task_to_contrib_scores[task_idx] =\
        np.sum(contribs_many_refs_func(
            task_idx=task_idx,
            input_data_sequences=fasta_sequences,
            num_refs_per_seq=num_refs_per_seq,
            batch_size=50,
            progress_update=4000,
        ),axis=2)[:,:,None]*onehot_data
    task_to_hyp_contrib_scores[task_idx] =\
        hypothetical_contribs_many_refs_func(
            task_idx=task_idx,
            input_data_sequences=fasta_sequences,
            num_refs_per_seq=num_refs_per_seq,
            batch_size=50,
            progress_update=4000,
        )

# Visualize the contributions and hypothetical contributions on a few sequences
from deeplift.visualization import viz_sequence

print("Scores for task 0, seq idx 0")
print("Actual contributions")
viz_sequence.plot_weights(task_to_contrib_scores[0][1])
print("Mean-normalized hypothetical contributions")
viz_sequence.plot_weights(task_to_hyp_contrib_scores[0][1])

# Save the importance scores
import h5py
import os

if (os.path.isfile("scores.h5")):
    !rm scores.h5
f = h5py.File("scores.h5",'w')
g = f.create_group("contrib_scores")
for task_idx in all_tasks:
    g.create_dataset("task"+str(task_idx), data=task_to_contrib_scores[task_idx])
g = f.create_group("hyp_contrib_scores")
for task_idx in all_tasks:
    g.create_dataset("task"+str(task_idx), data=task_to_hyp_contrib_scores[task_idx])
f.close()