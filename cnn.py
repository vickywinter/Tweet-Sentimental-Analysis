#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:28:40 2018

@author: vickywinter
"""

import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
#from keras.layers import Merge
from keras.layers import merge
import numpy as np


labels1=list(train['label'])
new_texts1 =list(train['tweet'])
MAX_SEQUENCE_LENGTH = 1000
 


def cnn(data, test_data):
    td=data
    labels=list(td['label'])
    new_texts=list(td['tweet'])
    
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100 
    VALIDATION_SPLIT = 0.2
    
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(new_texts)
    sequence = tokenizer.texts_to_sequences(new_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
 
    data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    labels1 = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape) 
    print('Shape of label tensor:', labels1.shape)
 
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices) 
    data = data[indices]
    labels_train = labels1[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
 
    x_train = data[:-nb_validation_samples]
    y_train = labels_train[:-nb_validation_samples]
 
    x_val = data[-nb_validation_samples:]
    y_val = labels_train[-nb_validation_samples:]
 
    print('Number of positive and negative reviews in traing and validation set ')
    print (y_train.sum(axis=0))
    print (y_val.sum(axis=0))  
 
    GLOVE_DIR = "/Users/vickywinter/Documents/NYC/Twitter/glove.6B"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8")
    for line in f:    
        values = line.split()
        word = values[0]       
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index)) 
 
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
 
    for word, i in word_index.items():        
        embedding_vector = embeddings_index.get(word) 
        if embedding_vector is not None:         # words not found in embedding index will be all-zeros. 
            embedding_matrix[i] = embedding_vector
 
    embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=True)
 
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') 
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences) 
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1) 
    l_pool2 = MaxPooling1D(5)(l_cov2) 
    l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2) 
    l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling 
    l_flat = Flatten()(l_pool3) 
    l_dense = Dense(128, activation='relu')(l_flat)
 
    preds = Dense(2, activation='softmax')(l_dense)    # was 2 instead 11
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
    print("model fitting - simplified convolutional neural network")
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_val, y_val),nb_epoch=1, batch_size=128)
    y_train=model.predict(data)
    

    td=test_data
    new_texts_test=list(td['tweet'])
    
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(new_texts_test)
    sequence_test = tokenizer.texts_to_sequences(new_texts_test)
    word_index_test = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index_test))
 
    data_test = pad_sequences(sequence_test, maxlen=MAX_SEQUENCE_LENGTH)


    guess_cnn=cnn1.predict(data_test)
    guess_cnn=pd.DataFrame(guess_cnn)
    guess_cnn['CNN_prediction']=[1 if x>0.5 else 0 for x in guess_cnn[1]]

    guess_cnn_train=pd.DataFrame(y_train)
    guess_cnn_train['CNN_prediction']=[1 if x>0.5 else 0 for x in guess_cnn_train[1]]
    
    return (guess_cnn,guess_cnn_train)
    
    #another model
'''    
    convs = []
    filter_sizes = [3,4,5]
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
 
    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)
 
    l_merge = Merge(mode='concat', concat_axis=1)(convs) 
    l_cov1= Conv1D(128, 5, activation='relu')(l_merge) 
    l_pool1 = MaxPooling1D(5)(l_cov1) 
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1) 
    l_pool2 = MaxPooling1D(30)(l_cov2) 
    l_flat = Flatten()(l_pool2) 
    l_dense = Dense(128, activation='relu')(l_flat) 
    preds = Dense(11, activation='softmax')(l_dense)     
  
    model2 = Model(sequence_input, preds) 
    model2.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

    print("model fitting - more complex convolutional neural network")
    model2.summary()
    model2.fit(x_train, y_train, validation_data=(x_val, y_val),nb_epoch=2, batch_size=50) 

    return (model,y_train)



cnn1,y_train=cnn(train_data)

td=test_data
new_texts_test=list(td['tweet'])
    
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100 
VALIDATION_SPLIT = 0.2
    
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(new_texts_test)
sequence_test = tokenizer.texts_to_sequences(new_texts_test)
word_index_test = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index_test))
 
data_test = pad_sequences(sequence_test, maxlen=MAX_SEQUENCE_LENGTH)


guess_cnn=cnn1.predict(data_test)
guess_cnn=pd.DataFrame(guess_cnn)
guess_cnn['CNN_prediction']=[1 if x>0.5 else 0 for x in guess_cnn[1]]

guess_cnn_train=pd.DataFrame(y_train)
guess_cnn_train['CNN_prediction']=[1 if x>0.5 else 0 for x in guess_cnn_train[1]]

# a more complex model

  
    labels1 = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape) 
    print('Shape of label tensor:', labels1.shape)
 
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices) 
    data = data[indices]
    labels_train = labels1[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
 
    x_train = data[:-nb_validation_samples]
    ynew = cnn1.predict(data_test)
    y_testtt=pd.DataFrame(ynew)
    y_testtt['label']=[1 if x>0.5 else 0 for x in y_testtt[1]]
    
ynew_train =pd.DataFrame(y_train)
ynew_train['label']=[1 if x>0.5 else 0 for x in ynew_train[1]]
guess_VM_train =pd.DataFrame(guess_VM_train)
guess_NB_train =pd.DataFrame(guess_NB_train)