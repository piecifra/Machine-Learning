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
confusion_matrix = defaultdict(int)
final_wrong = 0
final_correct = 0

TRAIN_DATASET_PATH = 'Dataset/train_dataset.jsonl'

latex = ''
K = 10
accuracy = float(0)


with open(TRAIN_DATASET_PATH, 'r') as f:
	l = f.read()
	lines = l.split('\n')[:-1]
	for l in lines:
		d = json.loads(l)
		op = []
		ops = []

		for o in d['instructions']:
			op.append(o.split()[0])
		
		for i in xrange(0,len(op),3):
			ops.append((tuple(op[i:i+3])))
		
		all_docs.append((d['compiler'],ops))
		for i in ops:
			V.add(i)
		C.add(d['compiler'])


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

	if i == 9:
		with open('compiler_pred_p.json', 'wb') as fp:
			pickle.dump(P, fp, protocol=pickle.HIGHEST_PROTOCOL)
		with open('compiler_pred_pj.json', 'wb') as fp:
			pickle.dump(Pj, fp, protocol=pickle.HIGHEST_PROTOCOL)	

	correct = 0
	wrong = 0
	ris = ''
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

		if i == 9:
			confusion_matrix[ris, corrct_class] += 1

		if ris == corrct_class:
			correct += 1
		else:
			wrong += 1 

	accuracy_i = float(correct)/(correct+wrong)
	accuracy += accuracy_i
	final_correct += correct
	final_wrong += wrong
	print('correct: ' + str(correct))
	print('wrong: ' + str(wrong))
	print('accuracy: ' + str(100*accuracy_i) + '%')
	print()

	latex += str(i+1) + '&' + str(100*accuracy_i)[:5] + '\\%&' + str(correct) + '&' + str(wrong)  + '\\\\' + '\n'


final_accuracy = float(accuracy)/K
print('Accuracy from ' + str(K) + '-Fold Cross validation: ' + str(100*accuracy) + '%')

confusion_matrix_array = []


if 0 == 1:
	#x->c2->correct
	#y->c1->ris
	for c1 in C:
		l = []
		for c2 in C:
			l.append(confusion_matrix[c1,c2])
		confusion_matrix_array.append(l)

	print('Printing confusion matrix for all Iteration')
	import seaborn as sn
	import pandas as pd
	import matplotlib.pyplot as plt

	df_cm = pd.DataFrame(confusion_matrix_array, index = C, columns = C)
	plt.figure(figsize = (10,7))
	sn.set(font_scale=1.4)#for label size
	sn.heatmap(df_cm, fmt='d', annot=True,annot_kws={"size": 14})# font size
	plt.show()

latex +=  '[1ex]\n\\hline\n'
latex += 'Final&' + str(100*final_accuracy)[:5] + '\\%&' + str(final_correct) + '&' + str(final_wrong) + '\\\\'

print('\\begin{table}[ht]\n\\centering')
print('\\caption{' + str(K) + ' + -fold cross validation w str(features)'.strip('[]').replace('\'','').replace('_', '\\_') + '}')
print('\\begin{tabular}{c c c c}')
print('\\hline\\hline')
print('Iteration & Accuracy & Correct & Wrong \\\\ [0.5ex]')
print latex
print('\\end{tabular}\n\\label{table:nonlin}\n\\end{table}')

