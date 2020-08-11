import numpy as np
import nltk, random
import torch
import pandas as pd

def get_sentences(fname):
	f = open(fname)
	lines = f.readlines()
	f.close()
	lines = [x.strip() for x in lines]
	return lines

def inplace_shuffle(a,b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a,b

def get_dataset_dict(language, label, split):
	prefix = 'data/'
	npy_fname = prefix + language+"."+split+"."+label+".vec.npy"
	np_matrix = np.load(npy_fname)
	data_dict = {}
	fname = prefix + language+"."+split+"."+label+".sent.txt"
	f = open(fname)
	lines = f.readlines()
	f.close()
	assert(len(lines) == len(np_matrix))
	for index in range(len(lines)):
		data_dict[lines[index].strip()] = np_matrix[index]
	return data_dict

def word_embedding_list(sentence, data_dict):
	return data_dict[sentence.strip()]

def word_embedding_list_withwords(sentence, data_dict):
	return data_dict[sentence.strip()], sentence.strip().split()


def create_one_training_example_xlmr(sentence, max_len, data_dict, dim=1024):
	sentence = " ".join(str(sentence).split()[:max_len])
	bag = list(word_embedding_list(sentence, data_dict))
	for i in range(max_len-len(bag)):
		bag.append(list(np.zeros(dim)))
	bag = bag[:max_len]
	return np.asarray(bag)

def word_embedding_list_xlmr(sentence, xlmr):
	word_emb_list = []
	for w in str(sentence).split():
		w = str(w)
		en_tokens = xlmr.encode(w)
		all_layers = xlmr.extract_features(en_tokens, return_all_hiddens=True)
		word_emb_sum = np.zeros(1024)
		for layer in all_layers:
			temp_sum = np.zeros(1024)
			for s in layer[0]:
				temp_sum += np.array(s.tolist())
			word_emb_sum += temp_sum/len(layer[0])
		word_emb_list.append(word_emb_sum/len(layer))
	return word_emb_list

def create_one_training_example(sentence, max_len, xlmr, dim=1024):

	sentence = " ".join(str(sentence).split()[:max_len])

	bag = word_embedding_list_xlmr(sentence, xlmr)

	for i in range(max_len-len(bag)):
		bag.append(list(np.zeros(dim)))

	bag = bag[:max_len]

	return np.asarray(bag)

def getXY(pos, neg):

	min_num = min(len(pos), len(neg))

	bag_pos = list(pos.values())[:min_num]
	bag_neg = list(neg.values())[:min_num]

	pos_labels = []
	for i in range(len(bag_pos)):
		pos_labels.append([1,0])

	neg_labels = []
	for i in range(len(bag_neg)):
		neg_labels.append([0,1])

	X_train = bag_pos + bag_neg
	Y_train = pos_labels + neg_labels
	
	if len(X_train) !=0 and len(Y_train) != 0:
		(X_train,Y_train) = inplace_shuffle(X_train,Y_train)

	Xoh = np.asarray(X_train)
	Yoh = np.asarray(Y_train)

	return Xoh, Yoh

def getXY_lc(pos1, neg1, pos2, neg2):

	min_num = min(len(pos1), len(neg1))
	bag_pos1 = list(pos1.values())[:min_num]
	bag_neg1 = list(neg1.values())[:min_num]
	bag1 = bag_pos1 + bag_neg1

	min_num = min(len(pos2), len(neg2))
	bag_pos2 = list(pos2.values())[:min_num]
	bag_neg2 = list(neg2.values())[:min_num]
	bag2 = bag_pos2 + bag_neg2

	pos_labels = []
	for i in range(len(bag1)):
		pos_labels.append([1,0])

	neg_labels = []
	for i in range(len(bag2)):
		neg_labels.append([0,1])


	X_train = bag1 + bag2
	Y_train = pos_labels + neg_labels
	
	if len(X_train) !=0 and len(Y_train) != 0:
		(X_train,Y_train) = inplace_shuffle(X_train,Y_train)

	Xoh = np.asarray(X_train)
	Yoh = np.asarray(Y_train)

	return Xoh, Yoh


def create_data4lstm_xlm_multilingual(train_language, test_language):

	source_pos_train_dict = get_dataset_dict(train_language, 'pos', 'train')
	source_neg_train_dict = get_dataset_dict(train_language, 'neg', 'train')
	source_pos_val_dict = get_dataset_dict(train_language, 'pos', 'val')
	source_neg_val_dict = get_dataset_dict(train_language, 'neg', 'val')
	target_pos_test_dict = get_dataset_dict(test_language, 'pos', 'test')
	target_neg_test_dict = get_dataset_dict(test_language, 'neg', 'test')

	# TRAIN
	X_train, Y_train = getXY(source_pos_train_dict, source_neg_train_dict)
	Y_train = np.reshape(Y_train, (Y_train.shape[0],1,2))

	# VAL
	X_val, Y_val = getXY(source_pos_val_dict, source_neg_val_dict)
	Y_val = np.reshape(Y_val, (Y_val.shape[0],1,2))

	# TEST
	X_test, Y_test = getXY(target_pos_test_dict, target_neg_test_dict)

	return X_train, Y_train, X_val, Y_val, X_test, Y_test

def create_train_multilingual_data(train_language):

	source_pos_train_dict = get_dataset_dict(train_language, 'pos', 'train')
	source_neg_train_dict = get_dataset_dict(train_language, 'neg', 'train')
	source_pos_val_dict = get_dataset_dict(train_language, 'pos', 'val')
	source_neg_val_dict = get_dataset_dict(train_language, 'neg', 'val')

	# TRAIN
	X_train, Y_train = getXY(source_pos_train_dict, source_neg_train_dict)
	Y_train = np.reshape(Y_train, (Y_train.shape[0],1,2))

	# VAL
	X_val, Y_val = getXY(source_pos_val_dict, source_neg_val_dict)
	Y_val = np.reshape(Y_val, (Y_val.shape[0],1,2))

	return X_train, Y_train, X_val, Y_val

def create_test_multilingual_data(test_language):

	target_pos_test_dict = get_dataset_dict(test_language, 'pos', 'test')
	target_neg_test_dict = get_dataset_dict(test_language, 'neg', 'test')

	# TEST
	X_test, Y_test = getXY(target_pos_test_dict, target_neg_test_dict)

	return X_test, Y_test

def create_train_multilingual_dual_data_4sub(train_language, test_language):

	source_pos_train_dict = get_dataset_dict(train_language, 'pos', 'train')
	source_neg_train_dict = get_dataset_dict(train_language, 'neg', 'train')
	source_pos_val_dict = get_dataset_dict(train_language, 'pos', 'val')
	source_neg_val_dict = get_dataset_dict(train_language, 'neg', 'val')

	target_pos_train_dict = get_dataset_dict(test_language, 'pos', 'train')
	target_neg_train_dict = get_dataset_dict(test_language, 'neg', 'train')
	target_pos_val_dict = get_dataset_dict(test_language, 'pos', 'val')
	target_neg_val_dict = get_dataset_dict(test_language, 'neg', 'val')

	# TRAIN
	X_train_o, Y_train_o = getXY(source_pos_train_dict, source_neg_train_dict)
	Y_train_o = np.reshape(Y_train_o, (Y_train_o.shape[0],1,2))

	# VAL
	X_val_o, Y_val_o = getXY(source_pos_val_dict, source_neg_val_dict)
	Y_val_o = np.reshape(Y_val_o, (Y_val_o.shape[0],1,2))

	# TRAIN
	X_train_m, Y_train_m = getXY_lc(source_pos_train_dict, source_neg_train_dict, target_pos_train_dict, target_neg_train_dict)
	Y_train_m = np.reshape(Y_train_m, (Y_train_m.shape[0],1,2))

	# VAL
	X_val_m, Y_val_m = getXY_lc(source_pos_val_dict, source_neg_val_dict, target_pos_val_dict, target_neg_val_dict)
	Y_val_m = np.reshape(Y_val_m, (Y_val_m.shape[0],1,2))

	return X_train_o, Y_train_o, X_val_o, Y_val_o, X_train_m, Y_train_m, X_val_m, Y_val_m
