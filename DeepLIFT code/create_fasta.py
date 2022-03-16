import numpy as np
import pandas as pd

index_path = './selected_index.txt'
file_path = './selected_fasta.fa'

if __name__ == "__main__":
    index = np.loadtxt(index_path).astype(np.int32)
    seq = pd.read_csv('./CNN_test_leaf.tsv', sep = '\t', header = 0,usecols = [3]).values
    seq = seq.flatten()
    selected_seq = seq[index]
    print(selected_seq)
    with open(file_path,'a') as file:
        for i in range(selected_seq.shape[0]):
            file.write(selected_seq[i]+'\n')
