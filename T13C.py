# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:15:23 2020

@author: Sami
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import random
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

folder = Path("/Users/Sami/Downloads/dogs-vs-cats/")
path_train = Path("/Users/Sami/Downloads/dogs-vs-cats/train/")
path_test = Path("/Users/Sami/Downloads/dogs-vs-cats/test1/")
path_array = []
label=[]

test_array = []
test_label = []

for file in os.listdir(path_train):
    path_array.append(os.path.join(path_train,file))
    if file.startswith("cat"):
        label.append('cat')
    elif file.startswith("dog"):
        label.append('dog') 
        
print(path_array[:5])
print(label[:5])

d = {'path': path_array, 'label': label}
X_train = pd.DataFrame(data=d)
# X_train.head()

for file in os.listdir(path_test):
    test_array.append(os.path.join(path_test,file))
   
        
test_d = {'path': test_array}
X_test = pd.DataFrame(data=test_d)


#%%
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(96,96,3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='categorical_crossentropy', metrics=['acc'])
model.summary()

#%%
dim = (96,96)
train_datagen = ImageDataGenerator(validation_split=0.08 ,rescale = 1.0/255.)
train_generator = train_datagen.flow_from_dataframe(dataframe= X_train,x_col='path',y_col='label',subset="training",batch_size=50,seed=42,shuffle=True, class_mode= 'categorical', target_size = dim)
valid_generator = train_datagen.flow_from_dataframe(dataframe= X_train,x_col='path',y_col='label',subset="validation",batch_size=50,seed=42,shuffle=True, class_mode= 'categorical', target_size = dim)


#%%

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

fitted_model = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=9,
                    verbose=1
)

#%%  
accuracy = fitted_model.history['acc']
#plt.imshow(X_train[1])
print(fitted_model.history)
plt.plot(range(len(accuracy)), accuracy, 'bo', label = 'accuracy')
plt.legend()


#%%
from tensorflow.keras.preprocessing import image

def catOrDog(img_number):
    test_image = image.load_img(X_train['path'][img_number], target_size=(96,96))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict_classes(x=test_image)[0]
    prob = model.predict_proba(x=test_image)
    fitted_model.history
    print(prob)
   
    if result == 1:
        prediction = ('Dog (Pr: ' + str( np.round( (prob[0][1]),3) ) +')')     
   
    else:
        prediction = ('Cat (Pr: ' + str( np.round( (prob[0][0]),3) ) +')')
    return prediction

def img_load(img_number):
    test_image = image.load_img(X_test['path'][img_number], target_size=(150,150))
    return test_image


fig, axs = plt.subplots(3, 3)
axs = axs.ravel()
for x in range(9):
    i = random.randint(0,len(X_test))
    print(i)
    axs[x].imshow(img_load(i))
    axs[x].text(150,-5, i, size=9, ha="right")
    axs[x].text(0,-5, catOrDog(i), size=9 )

    
img_list = {50,4163,1300,508,11490,5375}
fig2, axs = plt.subplots(2, 3)
axs = axs.ravel()
i=0
for img in img_list:
    print(i)
    axs[i].imshow(img_load(img))
    axs[i].text(150,-5, img , size=9, ha="right")
    axs[i].text(0,-5, catOrDog(img), size=9)
    i+=1


     




#%%
#model.save('T13_toinen_tapa.h5')

