# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:35:17 2020
Tehtävä 2 a
@author: Sami
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error



df = pd.read_csv('data/Google_Stock_Price.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Time']= df.apply(lambda row: len(df) - row.name, axis=1)
df['CloseFuture'] = df['Close'].shift(60)
df_test = df[:185]
df_train = df[185:]

X = np.array(df_train[['Time', 'Close']])
# X = X.reshape(-1,1) #tämä ohjeistettiin konsolissa
y = np.array(df_train['CloseFuture'])

model = linear_model.LinearRegression()

model.fit(X,y)
ennuste_train = model.predict(X)
df_train['Ennuste'] = ennuste_train

X_test = np.array(df_test[['Time','Close']])
# X_test = X_test.reshape(-1,1)
ennuste_test = model.predict(X_test)
df_test['Ennuste'] = ennuste_test

plt.scatter(df['Date'].values, df['Close'].values, color='black')
plt.plot((df_train['Date']+ pd.DateOffset(days=60)).values, df_train['Ennuste'].values, color='blue')
plt.plot((df_test['Date'] + pd.DateOffset(days=60)).values, df_test['Ennuste'].values, color='red')
plt.show()

df_validation = df_test.dropna()

error = mean_absolute_error(df_validation['CloseFuture'], df_validation['Ennuste'])
print("Ennusteen keskivirhe test datassa on %.f" % error)

error_train = mean_absolute_error(df_train['CloseFuture'], df_train['Ennuste'])
print("Ennusteen keskivirhe training datassa on %.f" % error_train)