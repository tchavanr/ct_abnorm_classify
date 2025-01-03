# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:13:24 2019

@author: trupti
"""
###########################################################################
#Aim: To define CNN architecture and preprocess the data
#Unet is considered as CNN architecture for segmentation of abnormalities, The architecture is modified to reduce training time.
#For pre-processing, data is normalized using mean and standard deviation.
###########################################################################
#load necessary packages
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
###########################################################################
#preprocess the dataset
#pre-processing = data-mean/dev
#input: ndarray of data to be preprcessed
#output: pre-processed data
def pre_process(data,data_mean,data_dev):
    data -= data_mean
    data /= data_dev
    return data
###########################################################################
#function to define the model architecture.
#Here unet architecture is used for segmentation of abnormalities
def model_def(img_rows, img_cols):
    inputs = Input((img_rows, img_cols,1),name = 'input_layer')

    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same',name = 'conv1_layer')(inputs)
    conv2 = Conv2D(4, (3, 3), activation='relu', padding='same',name = 'conv2_layer')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name = 'pool1_layer')(conv2)

    conv3 = Conv2D(8, (3, 3), activation='relu', padding='same',name = 'conv3_layer')(pool1)
    conv4 = Conv2D(8, (3, 3), activation='relu', padding='same',name = 'conv4_layer')(conv3)
#    pool2 = MaxPooling2D(pool_size=(2, 2),name = 'pool2_layer')(conv4)

#    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same',name = 'conv5_layer')(pool2)
#    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',name = 'conv6_layer')(conv5)
#    pool3 = MaxPooling2D(pool_size=(2, 2),name = 'pool3_layer')(conv6)

#    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same',name = 'conv7_layer')(pool3)
#    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same',name = 'conv8_layer')(conv7)
#
#    up1 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2),name = 'up1_layer')(conv8))
#    merge1 = concatenate([up1,conv6], axis = 3,name = 'merge1_layer')
#    conv9 = Conv2D(256, (3, 3), activation='relu', padding='same',name = 'conv9_layer')(merge1)
#    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same',name = 'conv10_layer')(conv9)

#    up2 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2),name = 'up2_layer')(conv10))
#    merge2 = concatenate([up2,conv4], axis = 3,name = 'merge2_layer')
#    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same',name = 'conv11_layer')(merge2)
#    conv12 = Conv2D(128, (3, 3), activation='relu', padding='same',name = 'conv12_layer')(conv11)

    up3 = Conv2D(4, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2),name = 'up3_layer')(conv4))#(conv12))
    merge3 = concatenate([up3,conv2], axis = 3,name = 'merge3_layer')
    conv13 = Conv2D(4, (3, 3), activation='relu', padding='same',name = 'conv13_layer')(merge3)
    conv14 = Conv2D(4, (3, 3), activation='relu', padding='same',name = 'conv14_layer')(conv13)
    
    conv15 = Conv2D(2, 3, activation = 'relu', padding = 'same',name = 'conv15_layer')(conv14)
    conv16 = Conv2D(1, (1, 1), activation='sigmoid',name = 'conv16_layer')(conv15)

    model = Model(inputs=[inputs], outputs=[conv16])
    return model