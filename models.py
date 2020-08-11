import keras
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import nltk, scipy, random
import pandas as pd
import sys, random, math, re, itertools
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk import word_tokenize
import operator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from numpy import linalg as LA
import math
from dataset_utils import *
from keras.layers import Embedding
from sklearn.metrics import confusion_matrix
import os.path
from os import path
######################## helper functions ########################
def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

######################## BILSTM ########################
def bilstm_model(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.4):

    X = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    A = Bidirectional(LSTM(units=n_a,dropout=drop))(X)
    A = Dense(10, activation = "relu",name='hidden')(A)
    out = Dense(out_dim, activation=softmax, name='classification')(A)
    model = Model(inputs=[X, s0, c0], outputs=out)
    return model

def train_bilstm(Xoh, Yoh, Tx, Ty, n_a=8, n_s=16, out_dim = 2, wv_dim=300, epochs=10, drop=0.4):

    model = bilstm_model(Tx, Ty, n_a, n_s, wv_dim, out_dim,drop=drop)

    #print(model.summary())

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'classification': 'categorical_crossentropy'},loss_weights={'classification': 1.0},optimizer=opt,metrics={'classification': 'accuracy'})
 
    s0 = np.zeros((len(Xoh), n_s))
    c0 = np.zeros((len(Xoh), n_s))
    outputs = list(Yoh.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    model.fit([Xoh, s0, c0], {'classification': outputs},batch_size=32,epochs=epochs,validation_split=0.15,shuffle=True, callbacks=callbacks)
    
    return model

def evaluate_bilstm(model, Xoh_test, Yoh_test, n_s=16):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred = []
    y_prob = []
    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        y_pred.append(np.argmax(prediction.squeeze()))
        if np.argmax(prediction.squeeze()) == 1:
            y_prob.append(np.max(prediction.squeeze()))
        else: 
            y_prob.append(1 - np.max(prediction.squeeze()))
    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))
    
    acc = accuracy_score(y_true, y_pred)
    
    return round(acc,4)

######################## BILSTM + ATTENTION ########################

def bilstm_attention_model(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.4):
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(softmax)
    dotor = Dot(axes = 1)
    post_activation_LSTM_cell = LSTM(n_s, return_state = True)

    X = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    attn_outputs = []
    a = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))(X)
    s_prev = repeator(s)
    concat = concatenator([a,s_prev])
    e = densor1(concat)
    e = Dropout(drop)(e)
    energies = densor2(e)
    alphas = activator(energies)

    context = dotor([alphas,a])

    s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])
    out_pre = Dense(n_s, activation = "relu")(s)
    output_layer = Dense(out_dim, activation=softmax, name='classification')(out_pre)

    word_attention = Lambda(lambda x: x[:, :,0,])(alphas)
    word_attention = Activation(None, name='word_attention')(word_attention)
    word_attention_copy = Lambda(lambda x: x[:, :,0,])(alphas)
    word_attention_copy = Activation(None, name='word_attention_copy')(word_attention_copy)

    model = Model(inputs=[X, s0, c0], outputs=[output_layer,word_attention,word_attention_copy])
    return model

def bilstm_attention_alignment_model(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.4):
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(softmax)
    dotor = Dot(axes = 1)

    post_activation_LSTM_cell = LSTM(n_s, return_state = True)
    bidirectional_layer = Bidirectional(LSTM(units=n_a, return_sequences=True,dropout=drop))

    X1 = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    a = bidirectional_layer(X1)
    s_prev = repeator(s)
    concat = concatenator([a,s_prev])
    e = densor1(concat)
    e = Dropout(drop)(e)
    energies = densor2(e)
    alphas1 = activator(energies)
    context = dotor([alphas1,a])
    s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])

    out_pre = Dense(n_s, activation = "relu")(s)
    o_classification = Dense(out_dim, activation=softmax, name='o_classification')(out_pre)

    repeator2 = RepeatVector(Tx)
    concatenator2 = Concatenate(axis=-1)
    densor3 = Dense(10, activation = "tanh")
    densor4 = Dense(1, activation = "relu")
    activator2 = Activation(softmax)
    dotor2 = Dot(axes = 1)

    post_activation_LSTM_cell2 = LSTM(n_s, return_state = True)
    bidirectional_layer2 = Bidirectional(LSTM(units=n_a, return_sequences=True,dropout=drop))

    X2 = Input(shape=(Tx, vocab_size))
    s = s0
    c = c0
    a = bidirectional_layer2(X2)
    s_prev = repeator2(s)
    concat = concatenator2([a,s_prev])
    e = densor3(concat)
    e = Dropout(drop)(e)
    energies = densor4(e)
    alphas2 = activator2(energies)
    context = dotor2([alphas2,a])
    s, _, c = post_activation_LSTM_cell2(context, initial_state = [s,c])

    out_pre = Dense(n_s, activation = "relu")(s)
    m_classification = Dense(out_dim, activation=softmax, name='m_classification')(out_pre)

    o_word_attention = Lambda(lambda x: x[:, :,0,])(alphas1)
    o_word_attention = Activation(None, name='o_word_attention')(o_word_attention)
    m_word_attention = Lambda(lambda x: x[:, :,0,])(alphas2)
    m_word_attention = Activation(None, name='m_word_attention')(m_word_attention)

    ###
    s = s0
    c = c0
    a = bidirectional_layer(X2)
    s_prev = repeator(s)
    concat = concatenator([a,s_prev])
    e = densor1(concat)
    e = Dropout(drop)(e)
    energies = densor2(e)
    alphas3 = activator(energies)

    s = s0
    c = c0
    a = bidirectional_layer2(X1)
    s_prev = repeator2(s)
    concat = concatenator2([a,s_prev])
    e = densor3(concat)
    e = Dropout(drop)(e)
    energies = densor4(e)
    alphas4 = activator2(energies)

    o_word_attention_m = Lambda(lambda x: x[:, :,0,])(alphas3)
    o_word_attention_m = Activation(None, name='o_word_attention_m')(o_word_attention_m)
    m_word_attention_o = Lambda(lambda x: x[:, :,0,])(alphas4)
    m_word_attention_o = Activation(None, name='m_word_attention_o')(m_word_attention_o)

    model = Model(inputs=[X1, X2, s0, c0], outputs=[o_classification,m_classification,o_word_attention,m_word_attention,o_word_attention_m,m_word_attention_o])
    return model

def create_model_attn_aggregation_jointlearning_sub(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.2, gamma_s=0.5, gamma_d=0.5):

    X = Input(shape=(Tx, vocab_size))
    X2 = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    ss = s0
    cc = c0

    dotor_sc = Dot(axes = 1)
    post_activation_LSTM_cell_sc = LSTM(n_s, return_state = True)
    pre_activation_LSTM_cell_sc = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))

    a_sc = pre_activation_LSTM_cell_sc(X)
    a_dc = pre_activation_LSTM_cell_sc(X2)

    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(softmax)

    s_prev = repeator(s)
    concat = concatenator([a_sc,s_prev])
    e = densor1(concat)
    e = Dropout(drop)(e)
    energies = densor2(e)
    alphas_sc = activator(energies)

    ss_prev = repeator(ss)
    concat = concatenator([a_dc,ss_prev])
    e = densor1(concat)
    e = Dropout(drop)(e)
    energies = densor2(e)
    alphas_dc = activator(energies)


    ####################################
    s = s0
    c = c0
    ss = s0
    cc = c0

    pre_activation_LSTM_cell = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))
    dotor = Dot(axes = 1)
    post_activation_LSTM_cell = LSTM(n_s, return_state = True)

    a = pre_activation_LSTM_cell(X2)
    aa = pre_activation_LSTM_cell(X)

    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(softmax)

    s_prev = repeator(s)
    concat = concatenator([a,s_prev])
    e = densor1(concat)
    e = Dropout(drop)(e)
    energies = densor2(e)
    alphas = activator(energies)

    ss_prev = repeator(ss)
    concat = concatenator([aa,ss_prev])
    e = densor1(concat)
    e = Dropout(drop)(e)
    energies = densor2(e)
    alphas0 = activator(energies)

    #normalize 
    alphas = Lambda(lambda x: K.l2_normalize(x,axis=1))(alphas)
    alphas_dc = Lambda(lambda x: K.l2_normalize(x,axis=1))(alphas_dc)
    #clip
    alphas = Lambda(lambda x: K.clip(x,-1,1))(alphas)
    alphas_dc_orig = Lambda(lambda x: K.clip(x,-1,1))(alphas_dc)
    alphas_dc = Lambda(lambda x: K.clip(x,0,1))(alphas_dc)
    #gamma
    alphas_dc = Lambda(lambda x: x * gamma_d)(alphas_dc)
    alphas_dc_agg = keras.layers.Subtract()([alphas,alphas_dc])
    alphas_dc_agg = Lambda(lambda x: K.l2_normalize(x,axis=1))(alphas_dc_agg)

    #alphas_dc_agg = Dense(10, activation = "tanh")(alphas_dc_agg)
    #alphas_dc_agg = Dense(1, activation = softmax)(alphas_dc_agg)

    context = dotor([alphas_dc_agg,a])
    s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])

    out_pre = Dense(n_s, activation = "relu")(s)
    domain_classifier = Dense(out_dim, activation=softmax, name='domain_classifier')(out_pre)

    word_attention_dc = Lambda(lambda x: x[:, :,0,])(alphas_dc_agg)
    word_attention_dc = Activation(None, name='word_attention_dc')(word_attention_dc)

    word_attention_dc_A = Lambda(lambda x: x[:, :,0,])(alphas)
    word_attention_dc_A = Activation(None, name='word_attention_dc_A')(word_attention_dc_A)
    word_attention_dc_B = Lambda(lambda x: x[:, :,0,])(alphas_dc_orig)
    word_attention_dc_B = Activation(None, name='word_attention_dc_B')(word_attention_dc_B)

    ########### sc

    #normalize
    alphas_sc = Lambda(lambda x: K.l2_normalize(x,axis=1))(alphas_sc)
    alphas0 = Lambda(lambda x: K.l2_normalize(x,axis=1))(alphas0)
    #clip
    alphas_sc = Lambda(lambda x: K.clip(x,-1,1))(alphas_sc)
    alphas0_orig = Lambda(lambda x: K.clip(x,-1,1))(alphas0)
    alphas0 = Lambda(lambda x: K.clip(x,0,1))(alphas0)
    #gamma
    alphas0 = Lambda(lambda x: x * gamma_s)(alphas0)
    #subtract
    alphas_agg = keras.layers.Subtract()([alphas_sc,alphas0])
    alphas_agg = Lambda(lambda x: K.l2_normalize(x,axis=1))(alphas_agg)

    #alphas_agg = Dense(10, activation = "tanh")(alphas_agg)
    #alphas_agg = Dense(1, activation = softmax)(alphas_agg)

    s = s0
    c = c0
    context = dotor_sc([alphas_agg,a_sc])
    s, _, c = post_activation_LSTM_cell_sc(context, initial_state = [s,c])

    out_pre = Dense(n_s, activation = "relu")(s)
    sentiment_classifier = Dense(out_dim, activation=softmax, name='sentiment_classifier')(out_pre)

    word_attention_sc = Lambda(lambda x: x[:, :,0,])(alphas_agg)
    word_attention_sc = Activation(None, name='word_attention_sc')(word_attention_sc)

    word_attention_sc_A = Lambda(lambda x: x[:, :,0,])(alphas_sc)
    word_attention_sc_A = Activation(None, name='word_attention_sc_A')(word_attention_sc_A)
    word_attention_sc_B = Lambda(lambda x: x[:, :,0,])(alphas0_orig)
    word_attention_sc_B = Activation(None, name='word_attention_sc_B')(word_attention_sc_B)

    model = Model(inputs=[X, X2, s0, c0], outputs=[sentiment_classifier,domain_classifier,word_attention_sc,word_attention_dc,word_attention_sc_A,word_attention_sc_B,word_attention_dc_A,word_attention_dc_B])
    return model

def custom_loss_mttri(layer1, layer2, ortho_loss_weight=0.01):
    def loss(y_true,y_pred):
        ce_loss = K.categorical_crossentropy(y_true,y_pred)
        x = layer1.output
        y = layer2.output
        #ortho_loss = K.mean(K.sum(x*y,axis=1) / (K.sqrt(K.sum(x*x,axis=1)) * K.sqrt(K.sum(y*y,axis=1))))
        #Frobenius norm (Ruder and Plank, 2018)
        ortho_loss = K.sum(K.square(K.dot(x,K.transpose(y))))
        return ce_loss + ortho_loss_weight*ortho_loss
    return loss

def train_bilstm_attention_jointalignment_sub(dual_lingual_training_data, Tx, Ty, n_a=8, n_s=16, out_dim = 2, wv_dim=1024,epochs=50, sc_alignment_weight = 0.01, dc_alignment_weight = 0.01, drop=0.2, dc_weight=0.1,gamma_s=0.01,gamma_d=0.01):

    X_train_o = dual_lingual_training_data[0]
    Y_train_o = dual_lingual_training_data[1]
    X_val_o = dual_lingual_training_data[2]
    Y_val_o = dual_lingual_training_data[3]
    X_train_m = dual_lingual_training_data[4]
    Y_train_m = dual_lingual_training_data[5]
    X_val_m = dual_lingual_training_data[6]
    Y_val_m = dual_lingual_training_data[7]

    min_len = min(len(X_train_o), len(X_train_m))
    X_train_o = X_train_o[:min_len]
    Y_train_o = Y_train_o[:min_len]
    X_train_m = X_train_m[:min_len]
    Y_train_m = Y_train_m[:min_len]

    min_len = min(len(X_val_o), len(X_val_m))
    X_val_o = X_val_o[:min_len]
    Y_val_o = Y_val_o[:min_len]
    X_val_m = X_val_m[:min_len]
    Y_val_m = Y_val_m[:min_len]

    model = create_model_attn_aggregation_jointlearning_sub(Tx, Ty, n_a, n_s, wv_dim, out_dim,drop=drop,gamma_s=gamma_s,gamma_d=gamma_d)

    #print(model.summary())

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)

    model.compile(loss={'sentiment_classifier': custom_loss_mttri(model.get_layer('word_attention_sc_A'), model.get_layer('word_attention_sc_B'), sc_alignment_weight), 'domain_classifier': custom_loss_mttri(model.get_layer('word_attention_dc_A'), model.get_layer('word_attention_dc_B'), dc_alignment_weight)},loss_weights={'sentiment_classifier': 1.0, 'domain_classifier': dc_weight},optimizer=opt,metrics={'sentiment_classifier': 'accuracy', 'domain_classifier': 'accuracy'})
    #else:
        #model.compile(loss={'sentiment_classifier': custom_loss_realignment_layers(model.get_layer('word_attention_sc'), model.get_layer('word_attention_sc_d'), alignment_weight), 'domain_classifier': custom_loss_realignment_layers(model.get_layer('word_attention_dc'), model.get_layer('word_attention_dc_d'), alignment_weight)},loss_weights={'sentiment_classifier': 1.0, 'domain_classifier': dc_weight},optimizer=opt,metrics={'sentiment_classifier': 'accuracy', 'domain_classifier': 'accuracy'})

    s0 = np.zeros((len(X_train_o), n_s))
    c0 = np.zeros((len(Y_train_o), n_s))
    outputs_o = list(Y_train_o.swapaxes(0,1))[0]
    outputs_m = list(Y_train_m.swapaxes(0,1))[0]

    s0_val = np.zeros((len(X_val_o), n_s))
    c0_val = np.zeros((len(Y_val_o), n_s))
    outputs_val_o = list(Y_val_o.swapaxes(0,1))[0]
    outputs_val_m = list(Y_val_o.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    model.fit([X_train_o, X_train_m, s0, c0], {'sentiment_classifier': outputs_o, 'domain_classifier': outputs_m},batch_size=32,epochs=epochs,validation_data=[[X_val_o, X_val_m, s0_val, c0_val],[outputs_val_o, outputs_val_m]], shuffle=True, callbacks=callbacks, verbose=1)
    
    return model

def get_added_pseudo_labelled(model, pseudo_dataset, Tx):
    source_pos_train_dict = get_dataset_dict(pseudo_dataset, 'pos', 'train')
    source_neg_train_dict = get_dataset_dict(pseudo_dataset, 'neg', 'train')
    source_pos_val_dict = get_dataset_dict(pseudo_dataset, 'pos', 'val')
    source_neg_val_dict = get_dataset_dict(pseudo_dataset, 'neg', 'val')

    # TRAIN
    pos = {}
    neg = {}

    for sentence in source_pos_train_dict.keys():
        y, prob = predict_sentence(model, sentence, source_pos_train_dict)
        if prob > 0.7 and y == 0:
            pos[sentence] = source_pos_train_dict[sentence]
        if prob > 0.7 and y == 1:
            neg[sentence] = source_pos_train_dict[sentence]
    for sentence in source_neg_train_dict.keys():
        y, prob = predict_sentence(model, sentence, source_neg_train_dict)
        if prob > 0.7 and y == 0:
            pos[sentence] = source_neg_train_dict[sentence]
        if prob > 0.7 and y == 1:
            neg[sentence] = source_neg_train_dict[sentence]

    X_train, Y_train = getXY(pos, neg)
    Y_train = np.reshape(Y_train, (Y_train.shape[0],1,2))

    # VAL
    pos_val = {}
    neg_val = {}

    for sentence in source_pos_val_dict.keys():
        y, prob = predict_sentence(model, sentence, source_pos_val_dict)
        if prob > 0.7 and y == 0:
            pos_val[sentence] = source_pos_val_dict[sentence]
        if prob > 0.7 and y == 1:
            neg_val[sentence] = source_pos_val_dict[sentence]
    for sentence in source_neg_val_dict.keys():
        y, prob = predict_sentence(model, sentence, source_neg_val_dict)
        if prob > 0.7 and y == 0:
            pos_val[sentence] = source_neg_val_dict[sentence]
        if prob > 0.7 and y == 1:
            neg_val[sentence] = source_neg_val_dict[sentence]

    X_val, Y_val = getXY(pos_val, neg_val)
    Y_val = np.reshape(Y_val, (Y_val.shape[0],1,2))

    return X_train, Y_train, X_val, Y_val


def train_bilstm_attention_jointalignment_sub_pseudo(pseudo_dataset, dual_lingual_training_data, Tx, Ty, n_a=8, n_s=16, out_dim = 2, wv_dim=1024,epochs=50, sc_alignment_weight = 0.01, dc_alignment_weight = 0.01, drop=0.2, dc_weight=0.1,gamma_s=0.01,gamma_d=0.01):

    X_train_o = dual_lingual_training_data[0]
    Y_train_o = dual_lingual_training_data[1]
    X_val_o = dual_lingual_training_data[2]
    Y_val_o = dual_lingual_training_data[3]
    X_train_m = dual_lingual_training_data[4]
    Y_train_m = dual_lingual_training_data[5]
    X_val_m = dual_lingual_training_data[6]
    Y_val_m = dual_lingual_training_data[7]

    min_len = min(len(X_train_o), len(X_train_m))
    X_train_o = X_train_o[:min_len]
    Y_train_o = Y_train_o[:min_len]
    X_train_m = X_train_m[:min_len]
    Y_train_m = Y_train_m[:min_len]

    min_len = min(len(X_val_o), len(X_val_m))
    X_val_o = X_val_o[:min_len]
    Y_val_o = Y_val_o[:min_len]
    X_val_m = X_val_m[:min_len]
    Y_val_m = Y_val_m[:min_len]

    model = create_model_attn_aggregation_jointlearning_sub(Tx, Ty, n_a, n_s, wv_dim, out_dim,drop=drop,gamma_s=gamma_s,gamma_d=gamma_d)

    #print(model.summary())

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)

    model.compile(loss={'sentiment_classifier': custom_loss_mttri(model.get_layer('word_attention_sc_A'), model.get_layer('word_attention_sc_B'), sc_alignment_weight), 'domain_classifier': custom_loss_mttri(model.get_layer('word_attention_dc_A'), model.get_layer('word_attention_dc_B'), dc_alignment_weight)},loss_weights={'sentiment_classifier': 1.0, 'domain_classifier': dc_weight},optimizer=opt,metrics={'sentiment_classifier': 'accuracy', 'domain_classifier': 'accuracy'})
    #else:
        #model.compile(loss={'sentiment_classifier': custom_loss_realignment_layers(model.get_layer('word_attention_sc'), model.get_layer('word_attention_sc_d'), alignment_weight), 'domain_classifier': custom_loss_realignment_layers(model.get_layer('word_attention_dc'), model.get_layer('word_attention_dc_d'), alignment_weight)},loss_weights={'sentiment_classifier': 1.0, 'domain_classifier': dc_weight},optimizer=opt,metrics={'sentiment_classifier': 'accuracy', 'domain_classifier': 'accuracy'})

    s0 = np.zeros((len(X_train_o), n_s))
    c0 = np.zeros((len(Y_train_o), n_s))
    outputs_o = list(Y_train_o.swapaxes(0,1))[0]
    outputs_m = list(Y_train_m.swapaxes(0,1))[0]

    s0_val = np.zeros((len(X_val_o), n_s))
    c0_val = np.zeros((len(Y_val_o), n_s))
    outputs_val_o = list(Y_val_o.swapaxes(0,1))[0]
    outputs_val_m = list(Y_val_o.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    model.fit([X_train_o, X_train_m, s0, c0], {'sentiment_classifier': outputs_o, 'domain_classifier': outputs_m},batch_size=32,epochs=epochs,validation_data=[[X_val_o, X_val_m, s0_val, c0_val],[outputs_val_o, outputs_val_m]], shuffle=True, callbacks=callbacks, verbose=1)
    
    # get pseudo labelled dataset: model, dataset
    X_train_pseudo, Y_train_pseudo, X_val_pseudo, Y_val_pseudo = get_added_pseudo_labelled(model, pseudo_dataset, Tx)

    X_train = list(X_train_o)
    Y_train = list(Y_train_o)
    X_train.extend(list(X_train_pseudo))
    Y_train.extend(list(Y_train_pseudo))

    X_val = list(X_val_o)
    Y_val = list(Y_val_o)
    X_val.extend(list(X_val_pseudo))
    Y_val.extend(list(Y_val_pseudo))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    print("dual model finished..")


    model = bilstm_attention_model(Tx, Ty, n_a, n_s, wv_dim, out_dim, drop=drop)

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'classification': 'categorical_crossentropy'},loss_weights={'classification': 1.0},optimizer=opt,metrics={'classification': 'accuracy'})

    s0 = np.zeros((len(X_train), n_s))
    c0 = np.zeros((len(Y_train), n_s))
    outputs = list(Y_train.swapaxes(0,1))[0]

    s0_val = np.zeros((len(X_val), n_s))
    c0_val = np.zeros((len(Y_val), n_s))
    outputs_val = list(Y_val.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    model.fit([X_train, s0, c0], {'classification': outputs},batch_size=32,epochs=epochs,validation_data=[[X_val, s0_val, c0_val],outputs_val], shuffle=True, callbacks=callbacks, verbose=1)
    
    return model

def train_bilstm_attention(X_train, Y_train, X_val, Y_val, Tx, Ty, fname=None, n_a=8, n_s=16, out_dim = 2, wv_dim=1024,epochs=10,drop=0.2):

    model = bilstm_attention_model(Tx, Ty, n_a, n_s, wv_dim, out_dim, drop=drop)

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'classification': 'categorical_crossentropy'},loss_weights={'classification': 1.0},optimizer=opt,metrics={'classification': 'accuracy'})

    s0 = np.zeros((len(X_train), n_s))
    c0 = np.zeros((len(Y_train), n_s))
    outputs = list(Y_train.swapaxes(0,1))[0]

    s0_val = np.zeros((len(X_val), n_s))
    c0_val = np.zeros((len(Y_val), n_s))
    outputs_val = list(Y_val.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    model.fit([X_train, s0, c0], {'classification': outputs},batch_size=32,epochs=epochs,validation_data=[[X_val, s0_val, c0_val],outputs_val], shuffle=True, callbacks=callbacks, verbose=1)
    
    return model

def evaluate_bilstm_attention(model, Xoh_test, Yoh_test, n_s=16):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred = []
    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        y_pred.append(np.argmax(prediction[0].squeeze()))

    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)
    print("TP", tp)
    print("F1: ", f1_score(y_true, y_pred, average='weighted'))
    '''

    return round(acc,4), round(f1,4)

############

def custom_loss_realignment(layer1, layer2, ortho_loss_weight=0.01):
    def loss(y_true,y_pred):
        x = layer1.output
        y = layer2.output
        ortho_loss = K.mean(K.sum(x*y,axis=1) / (K.sqrt(K.sum(x*x,axis=1)) * K.sqrt(K.sum(y*y,axis=1))))
        #Frobenius norm (Ruder and Plank, 2018)
        #ortho_loss = K.sum(K.square(K.dot(x,K.transpose(y))))
        return K.categorical_crossentropy(y_true,y_pred) + ortho_loss_weight*(1-ortho_loss)
    return loss

def custom_loss_realignment_v2(layer1, layer2, ortho_loss_weight=0.01):
    def loss(y_true,y_pred):
        ce_loss = K.categorical_crossentropy(y_true,y_pred)
        e = 1e-7
        x = layer1.output
        y = layer2.output
        assert(len(x.shape) == len(y.shape))
        assert(len(x.shape) == 2)
        assert(x.shape[-1] != 1 and y.shape[-1] != 1)
        #arg_sorted = K.argsort(-x)
        #w = K.flip(K.array(range(max(x.shape))))
        #x = x[arg_sorted]*w
        #y = y[arg_sorted]
        x_m = K.mean(x, axis=1, keepdims=True)
        y_m = K.mean(y, axis=1, keepdims=True)
        x_s = K.std(x, axis=1, keepdims=True)
        y_s = K.std(y, axis=1, keepdims=True)
        x = (x-x_m)/(x_s+e)
        y = (y-y_m)/(y_s+e)
        x = K.clip(x,0,10)
        y = K.clip(y,0,10)
        #helper = K.where(x > 0, 1, 0)
        helper = K.sign(x)
        helper = K.clip(helper,0,1)
        #x = x[x>0]
        #y = y[:len(x)]
        y = y*helper
        l = 0
        #if float(K.get_value(K.sum(K.sum(x)))) == 0 or float(K.get_value(K.sum(K.sum(y)))) == 0:
            #return 0
        if len(x.shape) == 2:
            l = K.mean(K.sum(x*y,axis=1) / ((K.sqrt(K.sum(x*x,axis=1)) * K.sqrt(K.sum(y*y,axis=1)))+e))
        else:
            print("ERROR in custom loss")
        #if K.get_value(l)==0:
            #return 0
        return ce_loss + ortho_loss_weight*(1-l)
    return loss

def train_bilstm_attention_att_alignment(dual_training_data, Tx, Ty, alignment_weight=0.1, fname=None, n_a=8, n_s=16, out_dim = 2, wv_dim=1024,epochs=10,drop=0.2):

    X_train_o = dual_training_data[0]
    Y_train_o = dual_training_data[1]
    X_val_o = dual_training_data[2]
    Y_val_o = dual_training_data[3]
    X_train_m = dual_training_data[4]
    Y_train_m = dual_training_data[5]
    X_val_m = dual_training_data[6]
    Y_val_m = dual_training_data[7]

    model = bilstm_attention_alignment_model(Tx, Ty, n_a, n_s, wv_dim, out_dim, drop=drop)

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'o_classification': custom_loss_realignment(model.get_layer('o_word_attention'), model.get_layer('m_word_attention_o'), alignment_weight), 'm_classification': custom_loss_realignment(model.get_layer('m_word_attention'), model.get_layer('o_word_attention_m'), alignment_weight)},loss_weights={'o_classification': 1.0, 'm_classification': 1.0},optimizer=opt,metrics={'o_classification': 'accuracy', 'm_classification': 'accuracy'})

    s0 = np.zeros((len(X_train_o), n_s))
    c0 = np.zeros((len(Y_train_o), n_s))
    outputs_o = list(Y_train_o.swapaxes(0,1))[0]
    outputs_m = list(Y_train_m.swapaxes(0,1))[0]

    s0_val = np.zeros((len(X_val_o), n_s))
    c0_val = np.zeros((len(Y_val_o), n_s))
    outputs_val_o = list(Y_val_o.swapaxes(0,1))[0]
    outputs_val_m = list(Y_val_o.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    model.fit([X_train_o, X_train_m, s0, c0], {'o_classification': outputs_o, 'm_classification': outputs_m},batch_size=32,epochs=epochs,validation_data=[[X_val_o, X_val_m, s0_val, c0_val],[outputs_val_o, outputs_val_m]], shuffle=True, callbacks=callbacks, verbose=1)
    
    return model

def evaluate_bilstm_attention_att_alignment(model, Xoh_test, Yoh_test, n_s=16):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred_o = []
    y_pred_m = []
    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), np.expand_dims(sample, axis=0), s0, c0])
        y_pred_o.append(np.argmax(prediction[0].squeeze()))
        y_pred_m.append(np.argmax(prediction[1].squeeze()))

    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))

    acc_o = accuracy_score(y_true, y_pred_o)
    f1_o = f1_score(y_true, y_pred_o, average='weighted')
    acc_m = accuracy_score(y_true, y_pred_m)
    f1_m = f1_score(y_true, y_pred_m, average='weighted')

    return round(acc_o,4), round(f1_o,4), round(acc_m,4), round(f1_m,4)

def evaluate_bilstm_attention_att_alignment_sub(model, Xoh_test, Yoh_test, n_s=16):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred_o = []

    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), np.expand_dims(sample, axis=0), s0, c0])
        y_pred_o.append(np.argmax(prediction[0].squeeze()))

    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))

    acc_o = accuracy_score(y_true, y_pred_o)
    f1_o = f1_score(y_true, y_pred_o, average='weighted')

    return round(acc_o,4), round(f1_o,4)

def predict_sentence(model, sentence, data_dict, Tx=30, n_s=16):

    word_emb_list, word_list = word_embedding_list_withwords(sentence, data_dict)
    bag = []
    bag.extend(word_emb_list)
    for i in range(Tx-len(word_emb_list)):
        bag.append(list(np.zeros(1024)))
    bag = bag[:Tx]
    word_emb_list = np.array(bag)
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    prediction = model.predict([np.expand_dims(word_emb_list, axis=0),np.expand_dims(word_emb_list, axis=0), s0, c0])
    
    word_att1 = prediction[2].squeeze()
    y = np.argmax(prediction[0].squeeze())
    return np.argmax(prediction[0].squeeze()), prediction[0].squeeze()[y]


def get_attention_weights_xlmr(model, sentence, data_dict, Tx=30, n_s=16):
    word_emb_list, word_list = word_embedding_list_withwords(sentence, data_dict)
    bag = []
    bag.extend(word_emb_list)
    for i in range(Tx-len(word_emb_list)):
        bag.append(list(np.zeros(1024)))
    bag = bag[:Tx]
    word_emb_list = np.array(bag)
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    prediction = model.predict([np.expand_dims(word_emb_list, axis=0), s0, c0])
    word_att = prediction[1].squeeze()
    y = np.argmax(prediction[0].squeeze())
    word_att = list(word_att)
    ans = []
    for index in range(min(len(word_list),Tx)):
        ans.append((word_list[index],round(word_att[index],4)))
    return y, ans

def get_attention_weights_xlmr_dual(model, sentence, data_dict, Tx=30, n_s=16):
    word_emb_list, word_list = word_embedding_list_withwords(sentence, data_dict)
    bag = []
    bag.extend(word_emb_list)
    for i in range(Tx-len(word_emb_list)):
        bag.append(list(np.zeros(1024)))
    bag = bag[:Tx]
    word_emb_list = np.array(bag)
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    prediction = model.predict([np.expand_dims(word_emb_list, axis=0),np.expand_dims(word_emb_list, axis=0), s0, c0])
    
    word_att1 = prediction[2].squeeze()
    y1 = np.argmax(prediction[0].squeeze())
    word_att1 = list(word_att1)
    ans1 = []
    for index in range(min(len(word_list),Tx)):
        ans1.append((word_list[index],round(word_att1[index],4)))

    word_att2 = prediction[3].squeeze()
    y2 = np.argmax(prediction[1].squeeze())
    word_att2 = list(word_att2)
    ans2 = []
    for index in range(min(len(word_list),Tx)):
        ans2.append((word_list[index],round(word_att2[index],4)))

    return y1, y2, ans1, ans2

    
