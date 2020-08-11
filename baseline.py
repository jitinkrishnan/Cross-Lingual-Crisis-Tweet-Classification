from models import *
from dataset_utils import *
import sys
import keras

Tx = 30
Ty = 1
epochs = 100



def baseline_classifier(train_language, test_language, task):

	print("\nTRAIN SET: ", train_language)

	X_train, Y_train, X_val, Y_val = create_train_multilingual_data(train_language)

	print("Successfully created train/val splits: ", len(X_train), len(Y_train), len(X_val), len(Y_val))

	model = train_bilstm_attention(X_train, Y_train, X_val, Y_val, Tx, Ty, epochs=epochs)

	for test_set in [train_language, test_language]:

		print("\nTEST SET: ", test_set)

		X_test, Y_test = create_test_multilingual_data(test_set)
		print("\nSuccessfully created test splits: ", len(X_test), len(Y_test))

		acc, f1 = evaluate_bilstm_attention(model, X_test, Y_test)
		print("-Train:"+train_language + "-Test:" + test_set + "-Task:" + task)
		print("TEST Accuracy: ", acc)
		print("TEST F1: ", f1)


	################################ Attention ################################ 
	print("============= ATTENTION ================")

	for split in ['test', 'val', 'train']:

		print("split-", split)

		for language in [train_language, test_language]:
			pos_dict = get_dataset_dict(language, 'pos', split)
			neg_dict = get_dataset_dict(language, 'neg', split)

			print("TEST LANGUAGE: ", language)

			p_sentences = []
			p_ans = []
			n_sentences = []
			n_ans = []

			for sentence in list(pos_dict.keys()):
				y, ans = get_attention_weights_xlmr(model, sentence, pos_dict)
				if y == 1:
					p_sentences.append(sentence)
					p_ans.append(ans)
				else:
					n_sentences.append(sentence)
					n_ans.append(ans)

			print("----------- CORRECTLY CLASSIFIED POSITIVE EXAMPLES -----------")
			for i in range(len(p_sentences)):
				print("SENTENCE: ", p_sentences[i])
				print("ATTENTION: ", p_ans[i])
			print("----------- MISCLASSIFIED POSITIVE EXAMPLES -----------")
			for i in range(len(n_sentences)):
				print("SENTENCE: ", n_sentences[i])
				print("ATTENTION: ", n_ans[i])


			p_sentences = []
			p_ans = []
			n_sentences = []
			n_ans = []

			for sentence in list(neg_dict.keys()):
				y, ans = get_attention_weights_xlmr(model, sentence, neg_dict)
				if y == 0:
					p_sentences.append(sentence)
					p_ans.append(ans)
				else:
					n_sentences.append(sentence)
					n_ans.append(ans)

			print("----------- CORRECTLY CLASSIFIED NGEATIVE EXAMPLES -----------")
			for i in range(len(p_sentences)):
				print("SENTENCE: ", p_sentences[i])
				print("ATTENTION: ", p_ans[i])
			print("----------- MISCLASSIFIED NEGATIVE EXAMPLES -----------")
			for i in range(len(n_sentences)):
				print("SENTENCE: ", n_sentences[i])
				print("ATTENTION: ", n_ans[i])

	################################ ################################ 

baseline_classifier(sys.argv[1], sys.argv[2], 'request')
