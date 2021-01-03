# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 18:22:42 2020

@author: Sami
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/fruit_data.csv', sep=',', encoding='latin-1')

X = np.array(df[['mass', 'width', 'height', 'color_score']])

koodit = {'apple':0, 'lemon':1,'mandarin':2,'orange':3 }
df['fruit_name_koodi'] = df['fruit_name'].map(koodit)
y = np.array(df['fruit_name_koodi'])

scaler = preprocessing.StandardScaler() # skaalaa normaalijakautuneeksi. Vaatimus joissakin malleissa
X_scaled = scaler.fit_transform(X)

# Luodaan logistinen regressiomalli
model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg') #multi_class ja solver lisänä koska luokiteltavia muuttujia on neljä.
model.fit(X_scaled,y)
ennuste = model.predict(X_scaled)
print("Logistisen regressiomallin tarkkuus: " ,accuracy_score(y, ennuste))
df['LRennuste'] = ennuste

# Luodaan SVM malli
model_SVC = SVC()
model_SVC.fit(X_scaled,y)
ennuste_SVC = model_SVC.predict(X_scaled)
print("SVM mallin tarkkuus: ",accuracy_score(y, ennuste_SVC))
df['SVCennuste'] = ennuste_SVC

# Luodaan KNN malli 
model_KNN = KNeighborsClassifier()
model_KNN.fit(X_scaled,y)
ennuste_KNN = model_KNN.predict(X_scaled)
print("KNN mallin tarkkuus: ",accuracy_score(y, ennuste_KNN))
df['KNNennuste'] = ennuste_KNN