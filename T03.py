# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:59:41 2020

@author: Sami
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

a = {'x':[1,2,3,4,5], 'y':[1,2,1.3,3.75,2.25]}
df = pd.DataFrame(data = a)

df_test = df[4:]
df_train = df[:]
df_test  = df_test.append({'x':6}, ignore_index=True)

print(df_test)

X = np.array(df_train['x'])
X = X.reshape(-1,1) #tämä ohjeistettiin konsolissa
y = np.array(df_train['y'])

model = linear_model.LinearRegression()

model.fit(X,y)
ennuste_train = model.predict(X)
df_train['Ennuste'] = ennuste_train
print(model.predict(X))

X_test = np.array(df_test['x'])
X_test = X_test.reshape(-1,1) # vain jos yksiulotteinen taulukko
df_test['Ennuste'] = model.predict(X_test)

plt.scatter(df['x'].values, df['y'].values, color='black')
plt.plot((df_train['x']).values, df_train['Ennuste'].values, color='blue')
plt.plot((df_test['x']).values, df_test['Ennuste'].values, color='red')
plt.show()

# print("Ennuste muuttujan y arvoksi x:n arvolla 6 on", df_test['Ennuste'].values[4])

