import os
import cv2
import glob
import collections
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf




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

        img = cv2.resize(cv2.imread(PATH+str(classes)+'/'+samples),(112,112))
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img = cv2.equalizeHist(img)

        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        xdata[classes].append(equalized_img/255)


# массивный issue: картинки передавались в [0..255], а не в [0..1] и это была катастрофа

def show_image_samples(gen ):
    plt.figure(figsize=(20, 20))
    for i in range(len(gen)):
        plt.subplot(5, 5, i + 1)
        image=gen[i]
        plt.imshow(image)

        plt.title('', color='blue', fontsize=14)
        plt.axis('off')
    plt.show()



"""concatenating all classes and their respective labels"""
from sklearn.model_selection import train_test_split

Y = [0 for i in range(len(xdata[0]))] + [1 for i in range(len(xdata[1]))] + [2 for i in range(len(xdata[2]))] #concatenating both y data
X = xdata[0] + xdata[1] + xdata[2] #concatenating both x data

len(X), len(Y)
#sane length means correct processing

X = np.array(X)
Y = np.array(Y)

X.shape, Y.shape


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=[0]*1000+[1]*1000+[2]*1000)
print({i:ytest.tolist().count(i) for i in ytest.tolist()})
print({i:ytrain.tolist().count(i) for i in ytrain.tolist()})
print(xtrain[0].shape,xtrain[0].tolist())
show_image_samples(xtrain[:20])

del(X)
del(Y)
del(xdata)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,MaxPooling2D, BatchNormalization, Reshape, ReLU, add, GlobalAveragePooling2D, RandomRotation, RandomFlip
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow import compat
from keras.optimizers import Adam, SGD
from keras.regularizers import L1, L2, L1L2
from keras.utils import plot_model, model_to_dot
from tensorflow.keras import Sequential as Seq


data_augmentation = Seq([
  RandomFlip("horizontal_and_vertical"),
  RandomRotation(0.2),
])

# try:
#   sess = compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#   from tensorflow.python.client import device_lib
#   print(device_lib.list_local_devices())
# except:
#   from keras import backend as K
#   K.tensorflow_backend._get_available_gpus()

"""
v1
amount, size = 70, 6
model=Sequential()
model.add(Conv2D(70,(6,6),activation='relu',input_shape=(112,112,3)))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(70,(3,3),activation='relu'))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Dropout(0.5))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(35,(3,3),activation='relu'))
model.add(Dropout(0.5))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
print(model.summary())"""



#v2
model=Sequential()

model.add(Conv2D(518,(4,4),activation='relu',input_shape=(112,112,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(256+128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(64+16,(3,3),activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Conv2D(32,(2,2),activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Conv2D(8,(1,1),activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(20, activation='relu'))


model.add(Dense(3, activation='softmax'))


#v3
"""

         conv2d
         conv2d
    | |  maxpool (b1)
    | |
    | |> conv2d
    |    conv2d
| | |>   + (b2)
| |
| |>      conv2d
|         conv2d
|>     |  + (b3)
       |
       |> conv2d
          globalaveragepool
          dense(256)
          dropout
          dense(3)

"""
"""
inputs = keras.Input(shape=(112, 112, 3), name="img")

x = Conv2D(48, 3, activation="relu")(inputs)
x = Conv2D(48, 3, activation="relu")(x)
block_1_output = MaxPooling2D(3)(x)

x = Conv2D(48, 3, activation="relu")(block_1_output)
x = Conv2D(48, 3, activation="relu")(x)
rec_out = add([x, block_1_output])

for i in range(3):
  x = Conv2D(48, 3, activation="relu")(rec_out)
  x = Conv2D(48, 3, activation="relu")(x)
  rec_out = add([x, rec_out])

x = Conv2D(64, 3, activation="relu")(rec_out)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax')(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
"""

model.compile(optimizer = Adam(learning_rate=0.0015),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


print(model.summary())

h = model.fit(xtrain,ytrain,epochs=210,batch_size=32 #64
          ,validation_data = (xtest,ytest), shuffle=True)
import keras
from matplotlib import pyplot as plt
# regr = lambda history, name: np.polyfit(history.history[name]['epoch'], history.history[name][name], 1)

# def plotRegr(history, name, plt):
#   m, b = regr(history, name)
#   plt.plot(history.history[name]['epoch'], m*history.history[name]['epoch']+b)

history = h
plt.plot(history.history['accuracy'])
# plotRegr(history, 'accuracy', plt)
plt.plot(history.history['val_accuracy'])
# plotRegr(history, 'val_accuracy', plt)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train',  'val', 'rt', 'rv'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])



# plotRegr(history, 'loss', plt)
plt.plot(history.history['val_loss'])
# plotRegr(history, 'val_loss', plt)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'r', 'r'], loc='upper left')
plt.show()

from sklearn.metrics import classification_report

print('eval', model.evaluate(xtest,ytest))
model.evaluate(xtest,ytest)

from sklearn.metrics import confusion_matrix

#Predict
y_prediction = model.predict(xtest)

#Create confusion matrix and normalizes it over predicted (columns)
conf = confusion_matrix(ytest.argmax(axis=1), y_prediction.argmax(axis=1) , normalize='pred')

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt



df_cm = pd.DataFrame(conf, range(3), range(3))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.show()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
