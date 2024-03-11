# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import cv2 
import glob
import collections
import numpy as np
import matplotlib.pyplot as plt

os.listdir('train/')


first_class = []
PATH = 'train/'
xdata = collections.defaultdict(list)
for classes in [0,1,2]:
    ls =  os.listdir(PATH+str(classes))
    print(f"Processing images class: {classes}")
    for samples in ls:
        
        img = cv2.resize(cv2.imread(PATH+str(classes)+'/'+samples),(100,100))
        xdata[classes].append(img)

Y = [0 for i in range(len(xdata[0]))] + [1 for i in range(len(xdata[1]))] + [2 for i in range(len(xdata[2]))] #concatenating both y data
X = xdata[0] + xdata[1] + xdata[2] #concatenating both x data


print(len(X), len(Y))


X = np.array(X)
Y = np.array(Y)


print(X.shape, Y.shape)


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=0, shuffle=True)


del(X)
del(Y)
del(xdata)


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential


model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(100,100,3)))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='relu'))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.5))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Dropout(0.5))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
print(model.summary())


# this cell 27 



model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


model.fit(xtrain,ytrain,epochs=300,batch_size=64,validation_data = (xtest,ytest))


from sklearn.metrics import classification_report



model.evaluate(xtest,ytest)

