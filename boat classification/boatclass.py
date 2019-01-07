#!/usr/bin/env python3
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Malware')

parser.add_argument('--load-model', type=str, nargs='?',
                    help='Load model from path')

parser.add_argument('--save-model', type=str, nargs='?',
                    help='Save current model to path')


args = parser.parse_args()

LOAD_MODEL_PATH = args.load_model
SAVE_MODEL_PATH = args.save_model


import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import models
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D, AveragePooling2D
from shutil import copyfile
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from skimage import transform
from keras.models import load_model


import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

input_shape = (32, 32, 3)

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (32, 32, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def LeNet(input_shape, num_classes):
    
    print('\nLeNet model')
    model = models.Sequential()
    
    print('\tC1: Convolutional 6 kernels 5x5')
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
    print('\tS2: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    print('\tC3: Convolutional 16 kernels 5x5')
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    print('\tS4: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    print('\tC5: Convolutional 120 kernels 5x5')
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(Flatten())
    print('\tF6: Fully connected, 84 units')
    model.add(Dense(84, activation='tanh'))
    print('\tF7: Fully connected, 10 units')
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model


if 'test' not in os.listdir('.'):
    os.makedirs('./test/')
    gd = open('./sc5-test/ground_truth.txt', 'r')
    for l in gd.readlines():
    	name, cat = l.split(';')
    	cat = cat.strip().replace(' ', '').replace(':', '')
    	if cat not in os.listdir('./test'):
    		os.makedirs('./test/' + cat)
    	copyfile('./sc5-test/' + name, './test/' + cat + '/' + name)



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
	    rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        './sc5',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        './test',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')


model = LeNet(input_shape, 24)

if LOAD_MODEL_PATH is None:
	try:
		model.fit_generator(
	        	train_generator,
	        	steps_per_epoch=1000,
	        	epochs=50,
	        	validation_data=validation_generator,
	        	validation_steps=800)
	except KeyboardInterrupt:
		pass
else:
	model = load_model(LOAD_MODEL_PATH)

if SAVE_MODEL_PATH is not None:
	model.save(SAVE_MODEL_PATH)
try:
    score = model.evaluate_generator(generator=validation_generator, steps=10)
    print("Test loss: %f" %score[0])
    print("Test accuracy: %f" %score[1])
except KeyboardInterrupt:
	pass
    

label_map = (train_generator.class_indices)
inv_map = {v: k for k, v in label_map.items()}
print(inv_map)

C = label_map.keys()

cm = {}
for c1 in C:
	for c2 in C:
		cm[(c1,c2)] = 0
correct = 0

true_vector = []
pred_vector = []
wrong = 0
for p in os.listdir('./test.nosync'):
	if os.path.isdir('./test.nosync/' + p):
		for i in os.listdir('./test.nosync/' + p):
			if i.endswith('jpg') or i.endswith('jpeg'):
				image = load('./test.nosync' + '/' + p + '/' + i)
				y_prob = model.predict(image)
				y_classes = y_prob.argmax(axis=-1)
				true_vector.append(p)
				pred_vector.append(inv_map[y_classes[0]])
				try:
					if inv_map[y_classes[0]] == p:
						correct += 1
					else:
						wrong += 1
					cm[inv_map[y_classes[0]], p] += 1
				except:
					pass

print(correct)
print(wrong)

confusion_matrix_array = []
print(classification_report(true_vector, pred_vector))

#x->c2->predicted
#y->c1->real
for c1 in C:
	l = []
	for c2 in C:
		l.append(cm[c2,c1])
	confusion_matrix_array.append(l)

print('Printing confusion matrix')
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(confusion_matrix_array, index = C, columns = C)
plt.figure(figsize = (10,7))
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, fmt='d', annot=True,annot_kws={"size": 10})# font size
plt.show()

