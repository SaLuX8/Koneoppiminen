# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:55:58 2020

@author: Sami
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing

ennusteaika = 12 #kuukautta ennustetaan myyntiä 12kk eteenpäin
seqlength = 12 # syöteverktorin (historia) pituus kuukausina. Eli 12kk myynnit annetaan syötteenä, vapaaval

df = pd.read_csv('data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df['Time'] = df.index

#%%
# stationaarinen "muutos"-aikasarja. HUOM trendi katoaa
df['PssLag'] = df['Passengers'].shift(1)
df['PssDiff']= df.apply(lambda row:
                          row['Passengers']-row['PssLag'], axis=1)
    
for i in range(1, seqlength):
    df['PssDiffLag'+str(i)] = df['PssDiff'].shift(i)
    
for i in range(1,ennusteaika+1):
    df['PssDiffFut'+str(i)] = df['PssDiff'].shift(-i)

df_train = df.iloc[:-2*ennusteaika]
df_train.dropna(inplace=True)
df_test = df.iloc[-2*ennusteaika:]

#%% Muuttujien valinta ja skaalaus
input_vars = ['PssDiff']
for i in range(1,seqlength):
    input_vars.append('PssDiffFut'+str(i))

output_vars = []
for i in range(1, ennusteaika+1):
    output_vars.append('PssDiffFut'+str(i))

scaler = preprocessing.StandardScaler()
scalero = preprocessing.StandardScaler()

X = np.array(df_train[input_vars])
X_scaled = scaler.fit_transform(X)
X_scaledLSTM = X_scaled.reshape(X.shape[0],seqlength,1)  # viimeinen ykkönen kertoo, että on yksi aikasarja josta ennustetaan
y = np.array(df_train[output_vars])
y_scaled = scalero.fit_transform(y)

X_test = np.array(df_test[input_vars])
X_testscaled = scaler.transform(X_test)
X_testscaledLSTM = X_testscaled.reshape(X_test.shape[0],seqlength,1) #LSTM vaatima muoto

#%%
# trendin mallinnus lineaarisella regressiolla
from sklearn import linear_model
modelLR = linear_model.LinearRegression()
XLR = df_train['Time'].values
XLR = XLR.reshape(-1,1)
yLR = df_train['Passengers'].values
yLR = yLR.reshape(-1,1)
modelLR.fit(XLR, yLR)
XLR_test = df_test['Time'].values
XLR_test = XLR_test.reshape(-1,1)
df_test['PassengerAvgPred'] = modelLR.predict(XLR_test)

#%%

slope = modelLR.coef_

#%% LSTM-verkon muodostus ja opetus
modelLSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(24, input_shape=(seqlength,1), # input vain yksi aikasarja, mutta voisi olla myös useampia
                         return_sequences=False), #palautetaanko kerroksen outputtiin kunkin LSTM solun piilotettu tila vai ei
    # tf.keras.layers.LSTM(24, return_sequences=False), voisi olla toinenkin kerros, jolloin return_seq edellisessä olisi true
    tf.keras.layers.Dense(ennusteaika)
    ])
modelLSTM.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mse', 
              metrics=['mae'])

modelLSTM.fit(X_scaledLSTM, y_scaled, epochs=200, batch_size=seqlength)

#%% Ennusteen määritys. huom ennuste = ennusteDiff+ trendi
ennusteDiff = scalero.inverse_transform(
    modelLSTM.predict(X_testscaledLSTM[ennusteaika-1].reshape(1,12,1)))
ennuste = np.zeros(13)
ennuste[0] = df_test['Passengers'][df_test.index[ennusteaika-1]]
for i in range(1,13):
    for j in range(1,13):
        ennuste[j] = ennuste[j-1]+ennusteDiff[0][j-1]+slope #ennuste=ed.kk myynti+muutos+vuositrendin ka muutos
ennuste = np.array(ennuste[1:])

#%% Luodaan ennusteen piirtämistä varten oma dataframe
df_pred = df_test[-12:]
df_pred['PssPred'] = ennuste
#%% Piirretään pyydetty kuvaaja
plt.plot(df['Month'].values, df['Passengers'].values, color='black', label='Actual sales(training)')
plt.plot(df_pred['Month'].values, df_pred['PssPred'], color='red', label='Prediction')

plt.grid()
plt.legend()
plt.show()
#%%
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(df_pred['Passengers'].values,
                          df_pred['PssPred'].values))

#%%







