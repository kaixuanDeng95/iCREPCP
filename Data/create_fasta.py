import numpy as np
import pandas as pd

data_path = './CNN_leaf_total.xlsx'
index_path = './selected_index.txt'
file_path = './selected_fasta.fa'

if __name__ == "__main__":
    index = np.loadtxt(index_path).astype(np.int32)
    seq = pd.read_excel(data_path, header = 0,usecols = [1,2]).values
    
    seq,label = seq[:,1].flatten(),seq[:,0].flatten()
    selected_seq,selected_label = seq[index],label[index]
    with open(file_path,'a') as file:
        for i in range(selected_seq.shape[0]):
            file.write('>'+selected_label[i]+'\n')
            file.write(selected_seq[i]+'\n')
