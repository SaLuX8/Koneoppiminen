
"""
Created on Thu Nov  5 19:41:32 2020

@author: Sami
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/Kysynta.csv', sep=';', encoding='latin-1')
df['Date'] = df['Päivä']
df['Time']= df.apply(lambda row: len(df) - row.name, axis=1)
df['CloseFuture'] = df['Kysyntä'].shift(60)
df_test = df[:240]
df_train = df[240:]

X = np.array(df_train[['Time']])
X = X.reshape(-1,1) 
y = np.array(df_train['CloseFuture'])

model = linear_model.LinearRegression()

model.fit(X,y)
ennuste_train = model.predict(X)
df_train['Ennuste'] = ennuste_train

X_test = np.array(df_test[['Time']])
X_test = X_test.reshape(-1,1)
ennuste_test = model.predict(X_test)
df_test['Ennuste'] = ennuste_test

plt.scatter(df['Date'].values, df['Kysyntä'].values, color='black')
plt.plot(df_train['Date'].values+60, df_train['Ennuste'].values, color='blue')
plt.plot(df_test['Date'].values+60, df_test['Ennuste'].values, color='red')
plt.show()

df_validation = df_test.dropna()

error = mean_absolute_error(df_validation['CloseFuture'], df_validation['Ennuste'])
print("Ennusteen keskivirhe test datassa on %.f" % error)

error_train = mean_absolute_error(df_train['CloseFuture'], df_train['Ennuste'])
print("Ennusteen keskivirhe training datassa on %.f" % error_train)