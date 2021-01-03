# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:32:59 2020

@author: Sami
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # haetaan kerasin valmis datasetti mnist

#%%
# tehdään x_train ja x_test muuttujien koosta yksiulotteinen taulukko kertomalla 28x28=784
x_train_flat=x_train.reshape(60000, 28, 28, 1) #nro 1 merkitsee kuvan mustavalkoiseksi. RGB kuvassa olisi kolme 28px taulukkoa
x_test_flat = x_test.reshape(10000, 28, 28 ,1)

x_train_flat = x_train_flat/255 # jokainen pikseli on välillä 0-255, joten jaetaan 255 ja saadaan luku välillä 0-1
x_test_flat = x_test_flat/255
y_test_orig = y_test

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#%%

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(30, kernel_size=5, activation='relu', input_shape=(28,28,1)), #2-ulotteinen konvoluutiokerros, kunkin ikkunan koko 5x5px. Ei laitettu strides, joten ikkunat menevät tässä limittäin.
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2), #ikkunan koko 2x2px ja siirretään eteenpäin 2px. Ikkunat eivät siis mene limittäin. 
    tf.keras.layers.Conv2D(15, kernel_size=5, activation='relu'), #15 filtteriä
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),  #konvoluutio osuus loppuu
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') #outputissa 10 numeroa ja luokittelevan neuroverkon output activation on softmax
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy']) 

model.fit(x_train_flat, y_train, validation_data=(x_test_flat, y_test), epochs=11, batch_size=100)

#%%
ennuste_test = model.predict(x_test_flat)
enn = pd.DataFrame(ennuste_test)
enn['max'] = enn.max(axis=1)
enn['ennuste'] = enn.idxmax(axis=1)
enn['oikea'] = y_test_orig
enn['tulos'] = enn['ennuste'] == enn['oikea']

#%%
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

model.fit(x_train_flat, y_train, validation_data=(x_test_flat, y_test), epochs=1, batch_size=100)

#%%
#model.save('mnistconvmodel.h5')