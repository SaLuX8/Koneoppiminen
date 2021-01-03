# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 18:35:09 2020

@author: Sami
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error


#%% Luodaan ennustemalli
df = pd.read_csv('data/Telco.csv', sep=';')
df.fillna(0, inplace=True)
df_test = df.sample(n = 100)
df_train = df.drop(df_test.index)

#%%
import seaborn as sn
import matplotlib.pyplot as plt
corrMatrix = df_train.corr()

#%% Training
#cols= ['address','tenure','age','ed','equip','callcard','longmon','equipmon','longten','tollten','wiremon', 'internet', 'ebill','logequi', 'wireless','voice','pager','logcard']
#cols = ['tenure','age','marital','address','ed','income','employ','retire','region','reside','tollfree','equip','callcard','longmon','equipmon','cardmon','longten','tollten','cardten', 'internet', 'ebill','logequi']
cols = ['tenure','age','address','ed','equip','ebill','internet','employ','logcard','callcard']
#cols = df_train.iloc[:,:-1]
#check = df_train[cols]
X_train = np.array(df_train[cols])

y_train = np.array(pd.get_dummies(df_train['churn']))

#%% Model
scaler = preprocessing.StandardScaler() # skaalaa normaalijakautuneeksi. Vaatimus joissakin malleissa
X_scaled = scaler.fit_transform(X_train)

model = tf.keras.Sequential([
tf.keras.layers.Dense(20, activation="sigmoid", input_shape=(X_scaled.shape[1],)),  #X_scaled.shape[1] arvo on 4 
tf.keras.layers.Dense(20, activation="relu"),
tf.keras.layers.Dense(10, activation="relu"),
tf.keras.layers.Dense(2, activation="softmax") # luokittelevassa neuroverkossa täytyy olla yhtä monta output muuttujaa kuin on output luokkia eli tässä 4
]) # luokittelevassa neuroverkossa outputkerroksen activointifunktioksi määritellään softmax

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy']) 

model.fit(X_scaled, y_train, epochs=50, batch_size=1)

ennuste = model.predict(X_scaled)
df_train['Ennuste'] = np.round(ennuste[:,1],0).astype(int)

#%% TEST

X_test = np.array(df_test[cols])
X_test_scaled = scaler.fit_transform(X_test)
ennuste_test = model.predict(X_test_scaled)
df_test['Estimated churn'] = np.round(ennuste_test[:,1],0).astype(int)
df_test['churn risk'] = np.round(ennuste_test[:,1],3)
y_test = np.array(df_test['Estimated churn'])

from sklearn.metrics import accuracy_score
print("Test datan tarkkuus: ", accuracy_score(y_test, df_test['churn']))
print("Training datan tarkkuus: ", np.round(accuracy_score(df_train['churn'], df_train['Ennuste']),3))
sample = df_test[['churn','churn risk']].sample(20)

#%%
# model.save('T20_telco_60epoc.h5')

#%%
#model.fit(X_scaled, y_train, epochs=50, batch_size=1)
