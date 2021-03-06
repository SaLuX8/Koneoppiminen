# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:04:19 2020
Tehtävä 4
@author: Sami
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn import preprocessing

df = pd.read_csv('data/Google_Stock_Price.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Time']= df.apply(lambda row: len(df) - row.name, axis=1)
df['CloseFuture'] = df['Close'].shift(30)
df_test = df[:185]
df_train = df[185:]

X = np.array(df_train[['Time', 'Close']])
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = np.array(df_train['CloseFuture'])

model = tf.keras.Sequential([
tf.keras.layers.Dense(10, activation="relu", input_shape=(2,)), #10 neuronia hidden layerillä, 2 input muuttujaa, bias termi rectified linear funktio
tf.keras.layers.Dense(10, activation="relu"), #toinen piilotettu kerros
tf.keras.layers.Dense(1)])  # mallin output kerros

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mse', # numeerinen välimatka asteikollinen -> mse = mean square error, jos luokitteluarvollinen output-muuttuja categorical_crossentropy
              metrics=['mae'])  #mae, koska numeerinen mean acss error, jos luokittelu -> accuracy
# kutsutaan myös back propagation algoritm

model.fit(X_scaled, y, epochs = 100, batch_size = 10) # epochs: montako kertaa käydään läpi training data, batch_size: monenkodatarivin jälkeen päivitetään painokertoimia
ennuste_train = model.predict(X_scaled)
df_train['Ennuste'] = ennuste_train

X_test = np.array(df_test[['Time','Close']])
X_testscaled = scaler.transform(X_test)
ennuste_test = model.predict(X_testscaled)
df_test['Ennuste'] = ennuste_test

plt.scatter(df['Date'].values, df['Close'].values, color='black')
plt.plot((df_train['Date']+ pd.DateOffset(days=30)).values, df_train['Ennuste'].values, color='blue')
plt.plot((df_test['Date'] + pd.DateOffset(days=30)).values, df_test['Ennuste'].values, color='red')
plt.show()

df_validation = df_test.dropna()

error = mean_absolute_error(df_validation['CloseFuture'], df_validation['Ennuste'])
print("Ennusteen keskivirhe test datassa on %.f" % error)