# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:19:54 2020
@author: Sami
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn import preprocessing

df = pd.read_csv('data/Kysynta.csv', sep=';', encoding='latin-1')

df_test = df[300:]
df_train = df[:300]

for i in range (301,350):
    df_test = df_test.append({'Päivä': i}, ignore_index=True)



# --------- Train ---------
X = np.array(df_train['Päivä'])
X = X.reshape(-1,1)
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

y = np.array(df_train['Kysyntä'])

model = tf.keras.Sequential([
tf.keras.layers.Dense(10, activation="tanh", input_shape=(1,)), 
tf.keras.layers.Dense(10, activation="tanh"), 
tf.keras.layers.Dense(10, activation="relu"), 
tf.keras.layers.Dense(1)]) 

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse', 
              metrics=['mae']) 

model.fit(X_scaled, y, epochs = 50, batch_size = 10, verbose = 2)
# epochs = kierroksia, batch_size=painokertoimien päivitysväli (datariviä)


ennuste_train = model.predict(X_scaled)
df_train['Ennuste'] = ennuste_train
# -------------------------


# --------- Test ---------
X_test = np.array(df_test[['Päivä']])
X_test = X_test.reshape(-1,1)
X_testscaled = scaler.transform(X_test)
ennuste_test = model.predict(X_testscaled)
df_test['Ennuste'] = ennuste_test
# -------------------------


plt.scatter(df['Päivä'].values, df['Kysyntä'].values, color='black', s=1)
plt.plot((df_train['Päivä']).values, df_train['Ennuste'].values, color='blue')
plt.plot((df_test['Päivä']).values, df_test['Ennuste'].values, color='red')
plt.show()

df_test_validation = df_test.dropna()
df_train_validation = df_train.dropna()

error_train = mean_absolute_error(df_train_validation['Kysyntä'], df_train_validation['Ennuste'])
print("Ennusteen keskivirhe training datassa on %.f" % error_train)

error = mean_absolute_error(df_test_validation['Kysyntä'], df_test_validation['Ennuste'])
print("Ennusteen keskivirhe test datassa on %.f" % error)

