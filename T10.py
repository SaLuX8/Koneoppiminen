# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 22:21:23 2020

@author: Sami
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/Titanic.csv', sep=',', encoding='latin-1')

df_test = df.sample(n = 200)
df_train = df.drop(df_test.index)

X = (pd.get_dummies(df_train[['Pclass', 'Sex', 'SibSp','Embarked']]))
X['Age'] = df_train['Age']
X['Age'] = X['Age'].fillna(0)
X['Parch'] = df_train['Parch']
X = np.array(X)

y = np.array(pd.get_dummies(df_train['Survived']))

scaler = preprocessing.StandardScaler() # skaalaa normaalijakautuneeksi. Vaatimus joissakin malleissa
X_scaled = scaler.fit_transform(X)

# ========================================================
model = tf.keras.Sequential([
tf.keras.layers.Dense(30, activation= tf.nn.relu , input_shape=(X_scaled.shape[1],)),  # input muuttujien lkm 
tf.keras.layers.Dense(30, activation= tf.nn.relu ), 
tf.keras.layers.Dense(2, activation=tf.nn.softmax) # luokittelevassa neuroverkossa täytyy olla yhtä monta output muuttujaa kuin on output luokkia eli tässä 4
]) # luokittelevassa neuroverkossa outputkerroksen activointifunktioksi määritellään softmax

model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy']) 

model.fit(X_scaled, y, epochs=20, batch_size=1)

ennuste = np.argmax(model.predict(X_scaled), axis=1)
df_train['Ennuste'] = ennuste

# --------- LR TEST -------------------
X_test = (pd.get_dummies(df_test[['Pclass', 'Sex', 'SibSp','Embarked']]))
X_test['Age'] = df_test['Age']
X_test['Age'] = X_test['Age'].fillna(0)
X_test['Parch'] = df_test['Parch']

X_test = np.array(X_test)
y_test2 = np.array(df_test['Survived'])
y_test = np.array(pd.get_dummies(df_test['Survived']))

X_testscaled = scaler.transform(X_test)
ennuste_test = np.argmax(model.predict(X_testscaled),axis=1)
df_test['Ennuste'] = ennuste_test
print("tarkkuus test datassa: ",round(accuracy_score(y_test2, ennuste_test)*100,2),"%")
# ----------------------------------

df_otos = df_test[['PassengerId','Survived','Ennuste']].sample(n=20)