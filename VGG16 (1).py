#!/usr/bin/env python
# coding: utf-8

# **download the libraries**

# In[1]:
import os
from random import shuffle
from tkinter import Image
from tkinter.filedialog import askdirectory

import cv2
import pandas as pd
import sklearn
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dense
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from keras import models as kermdls

#variables
from tqdm import keras

IMAGE_WIDTH=100
IMAGE_HEIGHT=100
# IMAGE_SIZE = [100,100] #for VGG16
IMAGE_CHANNELS=3

# read the data set

dir = askdirectory(title="Select a train data set folder")#'D:\\myWork\\4th year-myWork\\graduated project\\Project development\\model\\data set\\the last version\\new version of data set 14-3-2023 (web scrapping)\\data set - Copy'
classes = ['1-Caries','2-Dental plaque','3-Abscess','4-Gingivitis']
Data = []
labeles = []
for category in os.listdir(dir):
    newPath = os.path.join(dir,category)
    for img in os.listdir(newPath):
        img_path = os.path.join(newPath,img)
        print(img_path)
        if 'Thumbs.db' not in img_path:
            # print(img_path)feature.canny(image2)
            print(img_path)
            img = cv2.imread(img_path,1)
            img = cv2.resize(img,(224,224))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # blue_channel, green_channel, red_channel = cv2.split(img)
            #
            # blueEdges = cv2.Canny(blue_channel, threshold1=0, threshold2=255)
            # greenEdges = cv2.Canny(green_channel, threshold1=0, threshold2=255)
            # redEdges = cv2.Canny(red_channel, threshold1=0, threshold2=255)
            #
            # cannymerge = cv2.merge([blueEdges, greenEdges, redEdges])

            #semegent each channel in colored image and merge it
            blue_channel, green_channel, red_channel = cv2.split(img)
            print(blue_channel.shape)

            ret, bluethresh = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY +
                                            cv2.THRESH_OTSU)
            ret, greenthresh = cv2.threshold(green_channel, 0, 255, cv2.THRESH_BINARY +
                                             cv2.THRESH_OTSU)
            ret, redthresh = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY +
                                           cv2.THRESH_OTSU)
            image_merge = cv2.merge([blue_channel, green_channel, red_channel])
            threshmerge = cv2.merge([bluethresh, greenthresh, redthresh])
            #
            # # end

            if img_path == r"D:\\myWork\\4th year-myWork\\graduated project\\Project development\\model\\data set\\the last version\\new version of data set 14-3-2023 (web scrapping)\\data set - Copy\\1-Caries\\40.jpg" or img_path==r"D:\\myWork\\4th year-myWork\\graduated project\\Project development\\model\\data set\\the last version\\new version of data set 14-3-2023 (web scrapping)\\data set - Copy\\2-Dental plaque\\69.jpg" or img_path==r"D:\\myWork\\4th year-myWork\\graduated project\\Project development\\model\\data set\\the last version\\new version of data set 14-3-2023 (web scrapping)\\data set - Copy\\3-Abscess\\abcess 1 (61).jpg" or img_path==r"D:\\myWork\\4th year-myWork\\graduated project\\Project development\\model\\data set\\the last version\\new version of data set 14-3-2023 (web scrapping)\\data set - Copy\\1-Caries\\4-Gingivitis\\(37).jpg":
                cv2.imshow("Show just one image to test",img)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img_array = np.array(img)
            # img_array = img_array.astype('float32') / 255.0
            # img_array = np.expand_dims(img_array, axis=0)
            Data.append((image.img_to_array(threshmerge)))
            labeles.append(classes.index(category))


#shuffle for data to make the computer learn more and understand don't save it
combined = list(zip(Data,labeles))
shuffle(combined)
Data[:],labeles[:] = zip(*combined)
X_train = np.array(Data)
Y_train = np.array(labeles)

print("---------------before np_utils.to_categorical-------------------")
for y in Y_train:
    print(y)

print("______________________")


Y_train = np_utils.to_categorical(Y_train) ###################uncoment here 1/7/2023
print("all data shape",np.shape(X_train))
print("all data label shape",np.shape(Y_train))

print("_____________after np_utils.to_categorical-------------------")
for y in Y_train:
    print(y)

print("______________________")



# Data Augmentation
# dataGen = ImageDataGenerator(rotation_range=20,width_shift_range=0.01,height_shift_range=0.01,horizontal_flip=True,vertical_flip=True)
# dataGen.fit(X_train)

#split data
X_train,x_test,Y_train,y_test= train_test_split(Data,labeles,test_size=0.000001)\

X_train = np.array(X_train)#######################################################hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
Y_train = np.array(Y_train)####################################################################here
x_test = np.array(x_test)
y_test = np.array(y_test)

Y_train = np_utils.to_categorical(Y_train)#########################################################################uncoment 1/7/23
y_test = np_utils.to_categorical(y_test)#########################################################################uncoment 1/7/2023

# hereeee uncomment first
# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


import tensorflow as tf
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


#checkpoint & save the model
file_path = 'modelTest10-ephocs50-val05_dataset-coloredPureimage-1-7-23.h5'
modelcheckpoint = ModelCheckpoint(file_path,monitor='accuracy',verbose=2,save_best_only=True,mode='max')
callBackList = [modelcheckpoint]
print("first xtrain  " , X_train.shape)
print("y_train shape before fit",Y_train.shape)
hist=model.fit(X_train,Y_train,epochs=20,validation_split=0.05,callbacks=callBackList)
model.summary()
print(model.evaluate(X_train,Y_train))

model.save('modelTest10-ephocs20-val05_dataset-RGBSegmentation-12-7-23.h5')############modelTest10-ephocs20-val05_dataset-RGBcanny-10-7-23.h5
# hereeee uncomment last

# # start prediction part
# print("saved model")
# mm = kermdls.load_model('modelTest10-ephocs20-val5_dataset-copy1-7-23.h5')############300000000
# # print("THE acccuracy of model ", mm.evaluate(x_test,y_test))
#
# y_pred = []
#
# print("x_test shape : ",x_test.shape)
# print(mm.predict(x_test))
#
# for image in x_test:
#     print("image shape ",image.shape)
#     image = np.array(image)
#     print("image shape after array : ", image.shape)
#     img = image.reshape(1, 100, 100, 3)
#     print("image shape after convert : ", image.shape)
#     image = np.expand_dims(image, axis=0)
#     print("image shape after convert : ", image.shape)
#     print("mm.predict(image)",mm.predict(image))
#     y_predi = np.argmax(mm.predict(image))#mm.predict(image)
#     print(y_predi)
#     y_pred.append(y_predi)
#
#
# print("_____________before np_utils.to_categorical_________________")
# for y in y_pred:
#     print(y)
#
# print("______________________")
# for y in y_test:
#     print(y)
#
#
# print("*************************************************************************")
#
# y_pred = np_utils.to_categorical(y_pred)######################uncoment 1/7/23
# # print("y_prediction : ",y_pred)
# # print("y_test : ",y_test)
#
# print("_____________after np_utils.to_categorical_________________")
# print("______________________y_predict____________________________")
# for y in y_pred:
#     print(y)
#
# print("_____________y_test_______________")
# for y in y_test:
#     print(y)
#
#
# print("*************************************************************************")
#
# y_test = np.expand_dims(y_test, axis=0)
#
# print("_____________after np.expand_dims(y_test, axis=0)_________________")
# print("______________________y_predict____________________________")
# for y in y_pred:
#     print(y)
#
# print("_____________y_test_______________")
# for y in y_test:
#     print(y)
#
#
# print("*************************************************************************")
#
#
#
# # y_pred = y_pred.flatten()
# y_test = y_test.flatten()
# print('y_pred',y_pred)
# print("y_test",y_test)
# # Create a Confusion Matrix
# print("the confusion matrix : ")
# labels = [0,1,2,3]
#
#
# print("_____________test the y label_________________")
# for y in y_pred:
#     print(y)
#
# print("______________________")
# for y in y_test:
#     print(y)


# confMat = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels)
# confMatDF = pd.DataFrame(confMat,index=labels,columns=labels)
# print(confMatDF )
#end prediction part