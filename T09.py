# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:26:01 2020

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

y = np.array(df_train['Survived'])

scaler = preprocessing.StandardScaler() # skaalaa normaalijakautuneeksi. Vaatimus joissakin malleissa
X_scaled = scaler.fit_transform(X)

# ========================================================
# ------ Luodaan logistinen regressiomalli ---------------
model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg') #multi_class ja solver lisänä koska luokiteltavia muuttujia on neljä.
model.fit(X_scaled,y)
ennuste = model.predict(X_scaled)
print("Logistisen regressiomallin tarkkuus (training data): " , round(accuracy_score(y, ennuste)*100,2),"%")
df_train['LRennuste'] = ennuste
# ----------------------------------------------------------

# --------- LR TEST -------------------
X_test = (pd.get_dummies(df_test[['Pclass', 'Sex', 'SibSp','Embarked']]))
X_test['Age'] = df_test['Age']
X_test['Age'] = X_test['Age'].fillna(0)
X_test['Parch'] = df_test['Parch']

X_test = np.array(X_test)
y_test = np.array(df_test['Survived'])

X_testscaled = scaler.transform(X_test)
ennuste_test = model.predict(X_testscaled)
df_test['LREnnuste'] = ennuste_test
print("Logistisen regressiomallin tarkkuus test datassa: " ,round(accuracy_score(y_test, ennuste_test)*100,2),"%")
# ----------------------------------

# ========================================================
# ----------- Luodaan SVM malli --------------------------
model_SVC = SVC()
model_SVC.fit(X_scaled,y)
ennuste_SVC = model_SVC.predict(X_scaled)
print("SVM mallin tarkkuus (training data): ", round(accuracy_score(y, ennuste_SVC)*100,2),"%")
df_train['SVMennuste'] = ennuste_SVC
# ---------------------------------------------------------

# --------- SVM TEST -------------------
X_test = (pd.get_dummies(df_test[['Pclass', 'Sex', 'SibSp','Embarked']]))
X_test['Age'] = df_test['Age']
X_test['Age'] = X_test['Age'].fillna(0)
X_test['Parch'] = df_test['Parch']

X_test = np.array(X_test)
y_test = np.array(df_test['Survived'])

X_testscaled = scaler.transform(X_test)
ennuste_test = model_SVC.predict(X_testscaled)
df_test['SVMEnnuste'] = ennuste_test
print("SVM tarkkuus test datassa: " ,round(accuracy_score(y_test, ennuste_test)*100,2),"%")
# ----------------------------------

# ========================================================
# ------------------- Luodaan KNN malli ------------------
model_KNN = KNeighborsClassifier()
model_KNN.fit(X_scaled,y)
ennuste_KNN = model_KNN.predict(X_scaled)
print("KNN mallin tarkkuus (training data): ", round(accuracy_score(y, ennuste_KNN)*100,2),"%")
df_train['KNNennuste'] = ennuste_KNN
# --------------------------------------------------------

# --------- KNN TEST -------------------
X_test = (pd.get_dummies(df_test[['Pclass', 'Sex', 'SibSp','Embarked']]))
X_test['Age'] = df_test['Age']
X_test['Age'] = X_test['Age'].fillna(0)
X_test['Parch'] = df_test['Parch']

X_test = np.array(X_test)
y_test = np.array(df_test['Survived'])

X_testscaled = scaler.transform(X_test)
ennuste_test = model_KNN.predict(X_testscaled)
df_test['KNNEnnuste'] = ennuste_test
print("KNN tarkkuus test datassa: " ,round(accuracy_score(y_test, ennuste_test)*100,2),"%")
# ----------------------------------

df_otos = df_test[['PassengerId','Survived','SVMEnnuste']].sample(n=20)