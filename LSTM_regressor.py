# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:36:38 2018

@author: Administrator
"""

import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,TimeDistributed,Dense
from keras.optimizers import Adam

BATCH_START=0
TIME_STEPS=20
BATCH_SIZE=50
INPUT_SIZE=1
OUTPUT_SIZE=1
CELL_SIZE=20
LR=0.006
def get_batch():
    global BATCH_START,TIME_STEPS
    #xs shape(50batch,20steps)
    xs=np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE,TIME_STEPS))/(10*np.pi)
    seq=np.sin(xs)
    res=np.cos(xs)
    BATCH_START+=TIME_STEPS
    #plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
    #plt.show()
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]
#get_batch()
model=Sequential()
#build a LSTM RNN
model.add(LSTM(
        batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),
        output_dim=CELL_SIZE,
        return_sequences=True, #默认是False,即最后一个时间步才会有输出，改成True之后是每个时间步都有一个输出
        stateful=True#默认是false,即每个batch之间是没有关系的,改成TRUE之后每个batch之间有关联
        ))
#add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam=Adam(LR)
model.compile(optimizer=adam,loss='mse')
print('Training-----')
for step in range(501):
    #data shape=(batch_num,steps,inputs/outputs)
    X_batch,Y_batch,xs=get_batch()
    cost=model.train_on_batch(X_batch,Y_batch)
    pred=model.predict(X_batch,BATCH_SIZE)
    plt.plot(xs[0,:],Y_batch[0].flatten(),'r',xs[0,:],pred.flatten()[:TIME_STEPS],'b--')
    plt.ylim((-1.2,1.2))
    plt.draw()
    plt.pause(0.5)
    if step%10==0:
        print('train cost:',cost)
