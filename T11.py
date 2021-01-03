# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:25:22 2020

@author: Sami
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # haetaan kerasin valmis datasetti mnist

#näin voidaan jakaa koodi blokkeihin, joita voi ajaa ctrl+enter
#%%  
plt.imshow(x_train[1], cmap='Greys')


#%%
# tehdään x_train ja x_test muuttujien koosta yksiulotteinen taulukko kertomalla 28x28=784
x_train_flat=x_train.reshape(60000, 784)
x_test_flat = x_test.reshape(10000, 784)

x_train_flat = x_train_flat/255 # jokainen pikseli on välillä 0-255, joten jaetaan 255 ja saadaan luku välillä 0-1
x_test_flat = x_test_flat/255

y_train = pd.get_dummies(y_train)
y_test_orig = y_test
y_test = pd.get_dummies(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation='relu', input_shape=(x_train_flat.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') #outputissa 10 numeroa ja luokittelevan neuroverkon output activation on softmax
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy']) 

model.fit(x_train_flat, y_train, validation_data=(x_test_flat, y_test), epochs=10, batch_size=100)

#%%
ennuste_test = model.predict(x_test_flat)

enn = pd.DataFrame(ennuste_test)
enn['max'] = enn.max(axis=1)
enn['ennuste'] = enn.idxmax(axis=1)
enn['oikea'] = y_test_orig
enn['tulos'] = enn['ennuste'] == enn['oikea']
#%%
import random

fig, axs = plt.subplots(2, 3)
axs = axs.ravel()
x=0
lista = [115,124,149,151,247,3893]
for i in lista:
    # i = random.randint(0,len(x_test))
    # number = y_test.loc[i][lambda x: x == 1].index[0]
    number = y_test_orig[i]
    #print(number)
    # pred_number = ennuste_test[i].max()
    pred_number = enn['ennuste'][i]
                      
    print(pred_number)
    axs[x].imshow(x_test[i])
    axs[x].text(27,-1, i, size=9, ha="right")
    axs[x].text(0,-1, ('pr',pred_number), size=9 )
    axs[x].text(12,-1, number, size=9 )
    axs[x].set_xticks([])
    axs[x].set_yticks([])
    x+=1
    
#%%
g=0
right = 0
wrong = 0

for i in y_test_orig:
    if i == enn['ennuste'][g]:
        right+=1
    else:
        wrong+=1
    g+=1
print(100-(wrong/right*100))