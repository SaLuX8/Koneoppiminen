# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 18:54:47 2020

@author: Sami
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('data/fruit_data.csv', sep=',', encoding='latin-1')

X = np.array(df[['mass', 'width', 'height', 'color_score']])


y = np.array(pd.get_dummies(df['fruit_name']))

scaler = preprocessing.StandardScaler() # skaalaa normaalijakautuneeksi. Vaatimus joissakin malleissa
X_scaled = scaler.fit_transform(X)


model = tf.keras.Sequential([
tf.keras.layers.Dense(30, activation="relu", input_shape=(X_scaled.shape[1],)),  #X_scaled.shape[1] arvo on 4 
tf.keras.layers.Dense(30, activation="relu"), 
tf.keras.layers.Dense(4, activation=tf.nn.softmax) # luokittelevassa neuroverkossa täytyy olla yhtä monta output muuttujaa kuin on output luokkia eli tässä 4
]) # luokittelevassa neuroverkossa outputkerroksen activointifunktioksi määritellään softmax

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy']) 

model.fit(X_scaled, y, epochs=50, batch_size=1)

ennuste = np.argmax(model.predict(X_scaled), axis=1)
df['Ennuste'] = ennuste