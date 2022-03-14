# Import the required modules:
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate,Flatten, Conv1D, AveragePooling1D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Enable GPU memory growth:

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Define a function to one-hot encode the DNA sequences (adapted from https://colab.research.google.com/drive/17E4h5aAOioh5DiTo7MZg4hpL6Z_0FyWr):

integer_encoder = LabelEncoder()  
one_hot_encoder = OneHotEncoder(categories='auto')

def one_hot_encoding(sequences, verbose = True): 
    one_hot_sequences = []
    if verbose:
        i = 0
        print('one-hot encoding in progress ...', flush = True)
    for sequence in sequences:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        one_hot_sequences.append(one_hot_encoded.toarray())
        if verbose:
            i += 1
            if i % 1000 == 0:
                print(i, 'sequences processed', flush = True, end = '\r')      
    if verbose:
        print('finished one-hot encoding:', i, 'sequences processed', flush = True)
    
    one_hot_sequences = np.stack(one_hot_sequences)

    return one_hot_sequences

# Define a dense block and transition layer, adapted from DenseNet (https://www.computer.org/csdl/proceedings-article/cvpr/2017/0457c261/12OmNBDQbld):

# A dense layer includes two convolution layers, a 1 Conv1D layer followed by a 3 Conv1D layer:
def dense_layer(x,growth_rate = 4,R = 1,kernel_size = 3,regular_rate = 1e-4,drop_rate = 0.2):
    x = Conv1D(filters = growth_rate*R,kernel_size = 1,strides = 1,padding = 'same',activation = 'relu',
               kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(x)
    x = Conv1D(filters = growth_rate,kernel_size = kernel_size,strides = 1,padding = 'same',activation = 'relu',
               kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(x)
    x = Dropout(drop_rate)(x)
    
    return x

# A dense block includes L dense layers, each layer takes all preceding feature-maps concatenated in depth as input:
def dense_block(x,layers = 4,growth_rate = 4,R = 1,kernel_size = 3,regular_rate = 1e-4,drop_rate = 0.2):
    for i in range(layers):
        conv = dense_layer(x,growth_rate = growth_rate,R = R,kernel_size = kernel_size,regular_rate = regular_rate,drop_rate = drop_rate)
        x = concatenate([x, conv], axis = 2)
        
    return x

# A transition layer includes a 1 Conv1D layer followed by a 2 average pooling layer:
def transition_layer(x,compression_rate = 0.5,regular_rate = 1e-4):
    filters = int(x.shape.as_list()[-1]*compression_rate)
    x = Conv1D(filters = filters,kernel_size = 1,strides = 1,padding = 'same',activation = 'relu',
               kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(x)
    x = AveragePooling1D(pool_size = 2)(x)
    
    return x

# Define a function to build the DenseNet model:
def dense_model(input_shape = (170, 4),regular_rate = 1e-4):
    inputs = Input(input_shape)
    x = Conv1D(filters = 72,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu',
               kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(inputs)
    x = Conv1D(filters = 72,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu',
               kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(x)
    x = AveragePooling1D(pool_size = 2)(x)
    x = dense_block(x,layers = 6,growth_rate = 12,R = 2)
    x = transition_layer(x)
    x = dense_block(x,layers = 12,growth_rate = 12,R = 4)
    x = transition_layer(x)
    x = dense_block(x,layers = 24,growth_rate = 12,R = 4)
    x = transition_layer(x)
    x = dense_block(x,layers = 16,growth_rate = 12,R = 4)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1,activation = 'linear',kernel_initializer = 'glorot_normal',kernel_regularizer = l2(regular_rate))(x)
    model = Model(inputs = inputs, outputs = outputs)
    
    return model

# Define a function to calculate the R2 score of each species of the model:
def calculate_rsquare(data):
    rsquare = [round(r2_score(data['enrichment'],data['predict']),2)]
    x = data['sp']
    for sp in ['At','Zm','Sb']:
        index = np.where(x == sp)
        true = data['enrichment'][index[0]]
        pred = data['pred'][index[0]]
        rsquare.append(round(r2_score(true,pred),2))
    corr = { 'species':['All','At','Zm','Sb'],
             'r2':rsquare }
    
    return pd.DataFrame(corr)


# Define training parameters:

batch_size = 128
epochs = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)


if __name__ == "__main__":
    # Load the training and test data of tobacco leaf and maize proto:
    data_test_leaf = pd.read_csv('./CNN_test_leaf.tsv', sep = '\t', header = 0)
    data_train_leaf = pd.read_csv('./CNN_train_leaf.tsv', sep = '\t', header = 0)
    data_test_proto = pd.read_csv('./CNN_test_proto.tsv', sep = '\t', header = 0)
    data_train_proto = pd.read_csv('./CNN_train_proto.tsv', sep = '\t', header = 0)

    # Encode the promoter sequences as One-hot matrix:
    train_sequences_leaf = one_hot_encoding(data_train_leaf['sequence'])
    test_sequences_leaf = one_hot_encoding(data_test_leaf['sequence'])
    train_sequences_proto = one_hot_encoding(data_train_proto['sequence'])
    test_sequences_proto = one_hot_encoding(data_test_proto['sequence'])

    # Convert the enrichment value to an array of the correct shape:
    train_enrichment_leaf = np.array(data_train_leaf['enrichment']).reshape(-1, 1)
    test_enrichment_leaf = np.array(data_test_leaf['enrichment']).reshape(-1, 1)
    train_enrichment_proto = np.array(data_train_proto['enrichment']).reshape(-1, 1)
    test_enrichment_proto = np.array(data_test_proto['enrichment']).reshape(-1, 1)

    # Define the training and test data of tobacco leaf:
    x_train,y_train,x_test,y_test = train_sequences_leaf,train_enrichment_leaf,test_sequences_leaf,test_enrichment_leaf
    
    # Train the DenseNet model for tobacco leaf system:
    print('tobacco leaf:')
    model = dense_model()
    model.compile(loss = 'mse', optimizer = Adam())
    model.fit(x_train, y_train, 
              epochs=epochs, 
              shuffle=True, 
              batch_size=batch_size,
              validation_split=0.1, 
              verbose=0, 
              callbacks=[early_stopping])
    # Save the model
    model.save('./model/cnn_densenet_leaf.h5')
    # Calculate the R2 score of the tobacco model:
    y_pred = model.predict(x_test)
    data_test_leaf['predict'] = y_pred.ravel()
    print('R square of the model in tobacco leaf system:')
    print(calculate_rsquare(data_test_leaf))
    print('\n')
    

    # Define the training and test data of maize proto:
    x_train,y_train,x_test,y_test = train_sequences_proto,train_enrichment_proto,test_sequences_proto,test_enrichment_proto

    # Train the DenseNet model for maize proto system:
    print('maize proto:')
    model = dense_model()
    model.compile(loss='mse', optimizer=Adam())
    model.fit(x_train, y_train, 
              epochs=epochs, 
              shuffle=True, 
              batch_size=batch_size,
              validation_split=0.1, 
              verbose=0, 
              callbacks=[early_stopping])
    y_pred = model.predict(x_test)
    # Save the model
    model.save('./model/cnn_densenet_proto.h5')
    # Calculate the R2 score of the maize proto model:
    y_pred = model.predict(x_test)
    data_test_proto['predict'] = y_pred.ravel()
    print('R square of the model in maize proto system:')
    print(calculate_rsquare(data_test_proto))
    print('\n')
        

