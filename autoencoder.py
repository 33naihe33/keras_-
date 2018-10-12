# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 20:10:08 2018

@author: Administrator
"""

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt

(x_train,_),(x_test,y_test)=mnist.load_data()
#data preprocessing
x_train=x_train.astype('float32')/255.-0.5 #minmax_normalized把值压缩到了-0.5到0.5之间
x_test=x_test.astype('float32')/255.-0.5
x_train=x_train.reshape((x_train.shape[0],-1))
x_test=x_test.reshape((x_test.shape[0],-1))
print(x_train.shape)
print(x_test.shape)
#编码器最终把数据编码为两个数字的形式
encoding_dim=2
#this is our input placeholder
input_img=Input(shape=(784,))
#encoder layers
encoded=Dense(128,activation='relu')(input_img)
encoded=Dense(64,activation='relu')(encoded)
encoded=Dense(10,activation='relu')(encoded)
encoder_output=Dense(encoding_dim)(encoded)
#decoder layers
decoded=Dense(10,activation='relu')(encoder_output)
decoded=Dense(64,activation='relu')(decoded)
decoded=Dense(128,activation='relu')(decoded)
decoded=Dense(784,activation='tanh')(decoded)
#construct the autoencoder model
autoencoder=Model(input=input_img,output=decoded)
#construct the encoder model for plotting
encoder=Model(input=input_img,output=encoder_output)
#compile autoencoder
autoencoder.compile(optimizer='adam',loss='mse')
#training
autoencoder.fit(x_train,x_train,
                nb_epoch=20,
                batch_size=256,
                shuffle=True)
#plotting
encoded_imgs=encoder.predict(x_test)
plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c=y_test)
plt.show()