# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:34:37 2018

@author: Administrator
"""

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop

#X shape(60000 28*28),y shape(10000,)
(X_train,y_train),(X_test,y_test)=mnist.load_data()

#data pre-processing
X_train=X_train.reshape(X_train.shape[0],-1)/255 #normalize
X_test=X_test.reshape(X_test.shape[0],-1)/255
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

#Another way to build your neural net
model=Sequential([
        Dense(32,input_dim=784),
        Activation('relu'),
        Dense(10),#只需要定义output_dim即可
        Activation('softmax')
        ])
#Another way to define your optimizer
rmsprop=RMSprop(lr=0.001,rho=0.9,epsilon=1e-8,decay=0.0)
#We add metrics to get more results you want to see
model.compile(
        optimizer=rmsprop,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )
print('Training-----')
#Another way to train the model
model.fit(X_train,y_train,nb_epoch=2,batch_size=32)

print('\nTesting-----')
#Evaluate the model with the metrics we defined earlier
loss,accuracy=model.evaluate(X_test,y_test)
print('test loss:',loss)
print('test accracy:',accuracy)