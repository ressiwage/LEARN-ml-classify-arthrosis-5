import os
import cv2
import glob
import collections
import numpy as np
import matplotlib.pyplot as plt

os.listdir('/content/drive/MyDrive/train/')

"""\<b> We can see that the number of samples per class are imbalanced. The quantity samples of class 0,1 and 2 are almost balanced, so we will be using these classes only. </b> </br>
Converting all images to size 320, 320
"""

first_class = []
PATH = '/content/drive/MyDrive/train/'
xdata = collections.defaultdict(list)
for classes in [0,1,2]:
    ls =  os.listdir(PATH+str(classes))
    print(f"Processing images class: {classes}")
    for samples in ls[:1000]:

        img = cv2.resize(cv2.imread(PATH+str(classes)+'/'+samples),(500,500))
        xdata[classes].append(img)

"""concatenating all classes and their respective labels"""

Y = [0 for i in range(len(xdata[0]))] + [1 for i in range(len(xdata[1]))] + [2 for i in range(len(xdata[2]))] #concatenating both y data
X = xdata[0] + xdata[1] + xdata[2] #concatenating both x data

len(X), len(Y)
#sane length means correct processing

X = np.array(X)
Y = np.array(Y)

X.shape, Y.shape

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
model.add(Conv2D(64,(15,15),activation='relu',input_shape=(500,500,3)))
#pooling layer
model.add(MaxPooling2D(3,3))
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

model2=Sequential()
model2.add(Conv2D(64,(3,3),activation='relu',input_shape=(100,100,3)))
#pooling layer
model2.add(MaxPooling2D(2,2))
model2.add(BatchNormalization())
model2.add(Conv2D(32,(3,3),activation='relu'))
#pooling layer
model2.add(MaxPooling2D(2,2))
model2.add(BatchNormalization())
model2.add(Conv2D(64,(3,3),activation='relu'))
model2.add(Dropout(0.5))
#pooling layer
model2.add(MaxPooling2D(2,2))
model2.add(BatchNormalization())
model2.add(Conv2D(32,(3,3),activation='relu'))
model2.add(Dropout(0.5))
#pooling layer
model2.add(MaxPooling2D(2,2))
model2.add(BatchNormalization())
model2.add(Flatten())
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(3, activation='softmax'))
print(model2.summary())
model2.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model2.fit(xtrain,ytrain,epochs=300,batch_size=64,validation_data = (xtest,ytest), shuffle=True, )
from sklearn.metrics import classification_report
model2.evaluate(xtest,ytest)