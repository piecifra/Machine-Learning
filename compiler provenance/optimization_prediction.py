import json
import random
import re
import cPickle as pickle

from itertools import groupby
from collections import defaultdict, Counter
from decimal import *

docs = defaultdict(list)
all_docs = []
Pj = defaultdict(float)
C = set()
V = set()


TRAIN_DATASET_PATH = 'Dataset/train_dataset.jsonl'
latex = ''
K = 10
final_accuracy = 0
final_precision = 0
final_recall = 0
final_f1score = 0


with open(TRAIN_DATASET_PATH, 'r') as f:
	l = f.read()
	lines = l.split('\n')[:-1]
	for l in lines:
		d = json.loads(l)
		op = []
		ops = []
		for o in d['instructions']:
			op.append(o.split()[0])
		
		for i in xrange(0,len(op),1):
			ops.append((tuple(op[i:i+5])))
		
		all_docs.append((d['opt'],ops))
		for i in ops:
			V.add(i)
		C.add(d['opt'])

random.shuffle(all_docs)

for d in all_docs:
	docs[d[0]].append(d[1])


for i in range(10):
	print '-------------------------'
	print '-------iteration ' + str(i) + '-------'
	words_in_docs = defaultdict(int)
	P = defaultdict(tuple)

	test_set_start = int(len(all_docs)*i/K)
	test_set_end = int(len(all_docs)*(i+1)/K)

	print 'Start ' + str(test_set_start)
	print 'End ' + str(test_set_end)


	test_set = all_docs[test_set_start:test_set_end]
	dataset_learning = all_docs[:test_set_start] + all_docs[test_set_end:]

	for c in C:
			tj = len(docs[c])
			Pj[c] = float(tj)/len(dataset_learning)
			words_in_docs = defaultdict(int)
			TFj = 0
			for instruction in docs[c]:
				TFj += len(instruction)
				for ii in instruction:
					words_in_docs[ii] += 1
			for w in V:
				tij = words_in_docs[w]
				P[(w,c)] = float((tij + 1))/(TFj + len(V))

	if i == 0:
		with open('opt_pred_p.json', 'wb') as fp:
			pickle.dump(P, fp, protocol=pickle.HIGHEST_PROTOCOL)
		with open('opt_pred_pj.json', 'wb') as fp:
			pickle.dump(Pj, fp, protocol=pickle.HIGHEST_PROTOCOL)

	correct = 0
	wrong = 0
	ris = ''
	print(len(test_set))
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for f in test_set:
		max = Decimal(0)
		for c in C:
			p = Decimal(Pj[c])
			for instruction in f[1]:
				p = p * Decimal(P[(instruction,c)])
		

			if p > max:
				max = p
				ris = c

		corrct_class = f[0]
		c = 'H'
		if ris == corrct_class and corrct_class == c:
			TP += 1
		if ris != corrct_class and corrct_class == c:
			FN += 1
		if ris == corrct_class and corrct_class != c:
			TN += 1
		if ris != corrct_class and corrct_class != c:
			FP += 1


	print '---Considering class ' + c + '----'
	print('True positive ' + str(TP))
	print('True negative ' + str(TN))
	print('False positive ' + str(FP))
	print('False negative ' + str(FN))

	accuracy = float(TP+TN)/(TP+TN+FP+FN)
	recall = float(TP)/(TP+FN)
	precision = float(TP)/(TP+FP)
	if precision + recall != 0:
		f1score = 2*float(precision*recall)/(precision+recall)
	else:
		f1score = 0
	print
	print('Accuracy ' + str(100*accuracy) + '%')
	print('Recall ' + str(recall))
	print('Precision ' + str(precision))
	print('F1-Score ' + str(f1score))

	latex += str(i+1) + '&' + str(100*accuracy)[:5] + '\\%&' + str(recall)[:5] + '&' + str(precision)[:5] + '&' + str(f1score)[:5] + '\\\\' + '\n'

	final_accuracy += accuracy
	final_recall += recall
	final_precision += precision
	final_f1score += f1score


final_accuracy = final_accuracy/K
final_recall = final_recall/K
final_precision = final_precision/K
final_f1score = final_f1score/K

latex +=  '[1ex]\n\\hline\n'
latex += 'Final&' + str(100*final_accuracy)[:5] + '\\%&' + str(final_recall)[:5] + '&' + str(final_precision)[:5] + '&' + str(final_f1score)[:5] + '\\\\'


print('Final Accuracy ' + str(final_accuracy) + '%')
print('Final Recall ' + str(final_recall))
print('Final Precision ' + str(final_precision))
print('Final F1-Score ' + str(final_f1score))


print('Table for latex document')

print('\\begin{table}[ht]\n\\centering')
print('\\caption{10-fold cross validation considering class: High}')
print('\\begin{tabular}{c c c c c}')
print('\\hline\\hline')
print('Iteration & Accuracy & Recall & Precision & F1-Score \\\\ [0.5ex]')
print('\\hline')
print(latex)
print('\\end{tabular}\n\\label{table:nonlin}\n\\end{table}')





