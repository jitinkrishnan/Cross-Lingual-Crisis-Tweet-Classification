from models import *
from dataset_utils import *
import sys
import keras

Tx = 30

def getVec(pos, Tx, xlmr):

	bag_pos = []
	for text in pos:
		bag_pos.append(create_one_training_example(text, Tx, xlmr))

	return np.asarray(bag_pos)

def bilstm_attention_classifier(language, split, task):

	prefix = 'data/'

	xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
	xlmr.eval()

	fname = None

	if split == 'train':
		fname = 'Multi-Lingual/disaster_response_messages_training.csv'
	elif split == 'val':
		fname = 'Multi-Lingual/disaster_response_messages_validation.csv'
	elif split == 'test':
		fname = 'Multi-Lingual/disaster_response_messages_test.csv'

	df = pd.read_csv(fname)
	df = df.loc[(df['original'] != df['message']) & (len(str(df['original']).split()) > 2) & (len(str(df['message']).split()) > 2)  ]

	category = None

	if language == 'en':
		category = 'message'
	else:
		category = 'original'

	pos = list((df.loc[df[task] == 1])[category])
	neg = list((df.loc[df[task] == 0])[category])
	pos = [x for x in pos if (type(x) == str and len(x.split()) > 2)]
	neg = [x for x in neg if (type(x) == str and len(x.split()) > 2)]

	min_num = min(len(pos),len(neg))
	pos = pos[:min_num]
	neg = neg[:min_num]

	pos_vec = getVec(pos, Tx, xlmr)
	neg_vec = getVec(neg, Tx, xlmr)

	assert(len(pos_vec) == len(pos))
	assert(len(neg_vec) == len(neg))

	pos_npy_fname = prefix + language+"."+split+"."+"pos.vec.npy"
	neg_npy_fname = prefix + language+"."+split+"."+"neg.vec.npy"
	pos_sent_fname = prefix + language+"."+split+"."+"pos.sent.txt"
	neg_sent_fname = prefix + language+"."+split+"."+"neg.sent.txt"

	np.save(pos_npy_fname, pos_vec)
	np.save(neg_npy_fname, neg_vec)

	f = open(pos_sent_fname, 'w+')
	for x in pos:
		f.write(x.strip()+'\n')
	f.close()

	f = open(neg_sent_fname, 'w+')
	for x in neg:
		f.write(x.strip()+'\n')
	f.close()

bilstm_attention_classifier(sys.argv[1], sys.argv[2], 'request')

