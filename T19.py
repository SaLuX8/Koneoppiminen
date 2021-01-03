# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:22:40 2020

@author: Sami
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
#%% Luodaan ennustemalli
df = pd.read_csv('data/MachineData.csv', sep=';', decimal='.')

X = np.array(pd.get_dummies(df[['Machine ID','Team','Provider','Lifetime','PressureInd','MoistureInd','TemperatureInd']]))
y = np.array(pd.get_dummies(df['Broken']))

scaler = preprocessing.StandardScaler() # skaalaa normaalijakautuneeksi. Vaatimus joissakin malleissa
X_scaled = scaler.fit_transform(X)

model = tf.keras.Sequential([
tf.keras.layers.Dense(20, activation="relu", input_shape=(X_scaled.shape[1],)),  #X_scaled.shape[1] arvo on 4 
tf.keras.layers.Dense(20, activation="relu"), 
tf.keras.layers.Dense(2, activation=tf.nn.softmax) # luokittelevassa neuroverkossa täytyy olla yhtä monta output muuttujaa kuin on output luokkia eli tässä 4
]) # luokittelevassa neuroverkossa outputkerroksen activointifunktioksi määritellään softmax

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy']) 

model.fit(X_scaled, y, epochs=10, batch_size=1)
#%%
ennuste = model.predict(X_scaled)
df['Ennuste'] = np.round(ennuste[:,1],3)
df.sort_values(by=['Ennuste'], ascending=False, inplace=True)

#%%
rikkoutuvat = df[df['Broken']==0]
jees = rikkoutuvat.nlargest(10, 'Ennuste', keep='first')

top10 = jees[['Machine ID', 'Ennuste']]

#%%
sample20 = df.sample(20)

