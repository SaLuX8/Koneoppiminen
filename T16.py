# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:19:18 2020

@author: Sami
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:06:36 2020

@author: Sami
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('data/2dclusters.csv', sep=';', header=None, names=['A', 'B'])


#%%
X = np.array(df[['A','B']])

inertia = []
for i in range(1,19):
    model = KMeans(n_clusters=i)
    model.fit(X)
    inertia.append(model.inertia_)

plt.scatter(np.arange(1,19),inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#%%
model = KMeans(n_clusters=7)
model.fit(X)
labels = model.labels_
df['Label'] = labels


#%%
from mpl_toolkits.mplot3d import Axes3D

colors = {0:'red', 1:'blue', 2:'green', 3:'magenta', 4:'black', 5:'orange', 6:'yellow'}

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(0,7):
    x=df.loc[df['Label'] == i]['A'].values
    y=df.loc[df['Label'] == i]['B'].values
  
    ax.scatter(x,y,marker='o',s=40, color=colors[i],label='Class'+str(i+1))
ax.set_xlabel('X')
ax.set_ylabel('Y')

ax.legend()
plt.show()