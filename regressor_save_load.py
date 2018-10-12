# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 20:41:47 2018

@author: Administrator
"""

import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

#create some data
X=np.linspace(-1,1,200)
np.random.shuffle(X)
Y=0.5*X+2+np.random.normal(0,0.05,(200,))

X_train,Y_train=X[:160],Y[:160]
X_test,Y_test=X[160:],Y[160:]
#build a neural network from the 1st layer to the last layer
model=Sequential()
model.add(Dense(output_dim=1,input_dim=1))
#choose loss function and optimizing method
model.compile(loss='mse',optimizer='sgd')
#training
print("Training-----")
for step in range(301):
    cost=model.train_on_batch(X_train,Y_train) #train_on_batch的默认返回值即compile中的Loss
#save
print('test before save:',model.predict(X_test[0:2]))
model.save('my_model.h5')
del model # deletes the existing model
#load
model=load_model('my_model.h5')
print('test after load:',model.predict(X_test[0:2]))