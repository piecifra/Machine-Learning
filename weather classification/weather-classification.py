#!/usr/bin/env python3
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Weather classification')

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
from keras.regularizers import l2


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
    
    print('\tC1: Convolutional 3 kernels 3x3')
    model.add(Conv2D(3, kernel_size=(3, 3), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    print('\tS2: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    print('\tC3: Convolutional 8 kernels 3x3')
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='valid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    print('\tS4: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    print('\tC5: Convolutional 64 kernels 3x3')
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='valid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Flatten())
    print('\tF6: Fully connected, 50 units')
    model.add(Dense(50, activation='tanh'))
    print('\tF7: Fully connected, 10 units')
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model



batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1. / 255,\
    zoom_range=0.1,\
    rotation_range=10,\
    width_shift_range=0.1,\
    height_shift_range=0.1,\
    horizontal_flip=True,\
    vertical_flip=False)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    directory='dataset_train.nosync',
    target_size=(32, 32),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    directory='pierfrancesco_test.nosync',
    target_size=(32, 32),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)


model = LeNet(input_shape, 4)
model.summary()

if LOAD_MODEL_PATH is None:
	try:
		history = model.fit_generator(
	        	train_generator,
	        	steps_per_epoch=100,
	        	epochs=10,
	        	validation_data=validation_generator,
	        	validation_steps=100)
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

DATASET_TEST = './dataset_test.nosync'
true_vector = []
pred_vector = []
wrong = 0
for p in os.listdir(DATASET_TEST):
    if os.path.isdir(DATASET_TEST + '/' + p):
        for i in os.listdir(DATASET_TEST + '/' + p):
            if i.lower().endswith('jpg') or i.lower().endswith('jpeg'):
                image = load(DATASET_TEST + '/' + p + '/' + i)
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

# accuracy
try:
    score = model.evaluate(Xtest, Ytest)
    print("Test loss: %f" %score[0])
    print("Test accuracy: %f" %score[1])

    preds = model.predict(Xtest,verbose=1)
    Ypred = np.argmax(preds, axis=1)

    Ypred = keras.utils.to_categorical(Ypred, num_classes)

    print('%s' %str(Ypred.shape))
    print('%s' %str(Ytest.shape))

    print(classification_report(Ytest, Ypred, digits=3))
except:
    pass

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
sn.set(font_scale=2)#for label size
sn.heatmap(df_cm, fmt='d', annot=True,annot_kws={"size": 20})# font size
plt.show()


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
