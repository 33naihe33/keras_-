# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:17:57 2018

@author: Administrator
"""

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam

(X_train,y_train),(X_test,y_test)=mnist.load_data()
#data pre processing
X_train=X_train.reshape(-1,1,28,28) #(-1是sample的个数，1是Channel通道数)
X_test=X_test.reshape(-1,1,28,28)
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

#Another way to build your CNN
model=Sequential()
#Conv layer 1 output shape(32,28,28)
model.add(Convolution2D(
        nb_filter=32,
        nb_row=5,#filter的宽度
        nb_col=5,#filter的高度
        border_mode='same',#padding method
        input_shape=(1,28,28)
        ))
model.add(Activation('relu'))
#pooling layer 1 output shape(32,14,14)
model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        border_mode='same'
        ))
#Conv layer 2 output shape (64,14,14)
model.add(Convolution2D(64,5,5,border_mode='same'))
model.add(Activation('relu'))
#pooling layer 2  output shape(64,7,7,)
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
#Fully connected layer 1 input shape(64*7*7)=3136,output shape(1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
#FC layer 2 to shape(10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))
#Another way to define your optimizer
adam=Adam(lr=1e-4)
#we add metrics to ger more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training-----')
#Another way to train the model
model.fit(X_train,y_train,nb_epoch=2,batch_size=32)
print('\nTesting-----')
#Evaluate the model with the metrics we defined earlier
loss,accuracy=model.evaluate(X_test,y_test)
print('\ntest loss:',loss)
print('\ntest accuracy:',accuracy)