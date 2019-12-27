from keras import applications
from keras.models import Model, Input
import keras
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
import os
#from google.colab import drive
from keras.preprocessing.image import ImageDataGenerator
#drive.mount('/content/drive')
import sklearn.metrics 
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
                         Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,\
                         UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from keras import callbacks
import numpy as np
from keras.models import load_model

datadir = '.'
trainingset = datadir+'/dataset_train.nosync'
testset = datadir + '/dataset_test.nosync/'

batch_size = 32
input_shape = ()
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,\
    zoom_range=0.1,\
    rotation_range=10,\
    width_shift_range=0.1,\
    height_shift_range=0.1,\
    horizontal_flip=True,\
    vertical_flip=False)

train_generator = train_datagen.flow_from_directory(
    directory=trainingset,
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255)

test_generator = test_datagen.flow_from_directory(
    directory=testset,
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

num_samples = train_generator.n
num_classes = train_generator.num_classes
input_shape = train_generator.image_shape

classnames = [k for k,v in train_generator.class_indices.items()]

print("Image input %s" %str(input_shape))
print("Classes: %r" %classnames)

print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
print('Loaded %d test samples from %d classes.' %(test_generator.n,test_generator.num_classes))


def load_backbone_net(input_shape):
    
    # define input tensor
    input0 = Input(shape=input_shape)

    # load a pretrained model on imagenet without the final dense layer
    feature_extractor = applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input0)
    
    
    feature_extractor = feature_extractor.output
    feature_extractor = Model(input=input0, output=feature_extractor)
    optimizer = 'adam' #alternative 'SGD'

    feature_extractor.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return feature_extractor


def transferNet(feature_extractor, num_classes, output_layer_name, trainable_layers):
    
    # get the original input layer tensor
    input_t = feature_extractor.get_layer(index=0).input

    # set the feture extractor layers as non-trainable
    for idx,layer in enumerate(feature_extractor.layers):
      if layer.name in trainable_layers:
        layer.trainable = True
      else:
        layer.trainable = False

    # get the output tensor from a layer of the feature extractor
    output_extractor = feature_extractor.get_layer(name = output_layer_name).output
    
    #output_extractor = MaxPooling2D(pool_size=(4,4))(output_extractor)

    # flat the output of a Conv layer
    flatten = Flatten()(output_extractor) 
    flatten_norm = BatchNormalization()(flatten)

    # add a Dense layer
    dense = Dropout(0.4)(flatten_norm)
    dense = Dense(100, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    
    # add a Dense layer
    dense = Dropout(0.4)(dense)
    dense = Dense(50, activation='relu')(dense)
    dense = BatchNormalization()(dense)

    # add the final output layer
    dense = BatchNormalization()(dense)
    dense = Dense(num_classes, activation='softmax')(dense)
    

    model = Model(input=input_t, output=dense, name="transferNet")
    
    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

# load the pre-trained model
feature_extractor = load_backbone_net(input_shape)
feature_extractor.summary()


# choose the layer from which you can get the features (block5_pool the end, glob_pooling to get the pooled version of the output)
name_output_extractor = "block5_pool"
trainable_layers = ["block5_conv3"]

# build the transfer model
transfer_model = transferNet(feature_extractor, num_classes, name_output_extractor, trainable_layers)
transfer_model.summary()


# fit the transferNet on the training data
stopping = callbacks.EarlyStopping(monitor='val_acc', patience=3)

steps_per_epoch = train_generator.n//train_generator.batch_size
val_steps = test_generator.n//test_generator.batch_size+1

try:
    history = transfer_model.fit_generator(train_generator, epochs=50, verbose=1,\
                    steps_per_epoch=steps_per_epoch,\
                    validation_data=test_generator,\
                    validation_steps=val_steps)
except KeyboardInterrupt:
    pass

transfer_model.save('./model.h5')


#transfer_model = load_model('./model.h5')

val_steps=test_generator.n//test_generator.batch_size+1

test_generator = test_datagen.flow_from_directory(
    directory=testset,
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# predictions from transferNet
preds = transfer_model.predict_generator(test_generator,verbose=1,steps=val_steps)

try:
    score = transfer_model.evaluate_generator(generator=test_generator, steps=10)
    print("Test loss: %f" %score[0])
    print("Test accuracy: %f" %score[1])
except KeyboardInterrupt:
    pass

'''
try:
    score = transfer_model.evaluate_generator(generator=train_generator, steps=10)
    print("Dataset loss: %f" %score[0])
    print("Dataset accuracy: %f" %score[1])
except KeyboardInterrupt:
    pass
'''
Ypred = np.argmax(preds, axis=1)
Ytest = test_generator.classes  # shuffle=False in test_generator

cm = confusion_matrix(Ytest, Ypred)
print(cm)
print(classification_report(Ytest, Ypred, digits=3))

conf = [] # data structure for confusions: list of (i,j,cm[i][j])
for i in range(0,cm.shape[0]):
  for j in range(0,cm.shape[1]):
    if (i!=j and cm[i][j]>0):
      conf.append([i,j,cm[i][j]])

col=2
conf = np.array(conf)
conf = conf[np.argsort(-conf[:,col])]  # decreasing order by 3-rd column (i.e., cm[i][j])

print('%-16s     %-16s  \t%s \t%s ' %('True','Predicted','errors','err %'))
print('------------------------------------------------------------------')
for k in conf:
  print('%-16s ->  %-16s  \t%d \t%.2f %% ' %(classnames[k[0]],classnames[k[1]],k[2],k[2]*100.0/test_generator.n))

print('Printing confusion matrix')
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

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

label_map = (train_generator.class_indices)
inv_map = {v: k for k, v in label_map.items()}

C = label_map.keys()

df_cm = pd.DataFrame(cm, index = C, columns = C)
plt.figure(figsize = (10,7))
sn.set(font_scale=2)#for label size
sn.heatmap(df_cm, fmt='d', annot=True,annot_kws={"size": 20})# font size
plt.show()


  