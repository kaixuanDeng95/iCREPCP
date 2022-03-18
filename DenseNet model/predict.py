import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

model_path = './cnn_densenet_leaf1.h5'
file_path = './selected_fasta.fa'

integer_encoder = LabelEncoder()  
one_hot_encoder = OneHotEncoder(categories='auto')

def one_hot_encoding(sequences): 
    one_hot_sequences = []
    for sequence in sequences:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        one_hot_sequences.append(one_hot_encoded.toarray())
    one_hot_sequences = np.stack(one_hot_sequences)

    return one_hot_sequences

if __name__ == "__main__":
    with open(file_path) as f:
        seq = []
        while True:
            line = f.readline()
            if not line:
                break
            if not line[0] == '>':
                seq.append(line.strip())
    x = one_hot_encoding(seq)
    model = load_model(model_path)
    y_pred = model.predict(x)
    pred_table = { 'sequences':seq,
                   'predict enrichments':y_pred.ravel()}
    pred_table = pd.DataFrame(pred_table)
    print(pred_table)


