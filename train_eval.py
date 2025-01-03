# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:13:24 2019

@author: trupti
"""
###########################################################################
#Main file to run
#Aim: To trian the CNN for segmentation of abnormalities
#Training details: SGD is used for optimization. 
#learning rate=0.001, weight decay = 5*1e-4, momentum=0.9
#Training is carried for 5 epochs. This can be increased for improving the accuracy.
#batch_size is kept to 64.
#Steps:
#1. Load the train, val and test dataset
#2. define the network architecture
#3. Train the network
#4. Evaluate the network.
###########################################################################
#load necessary packages
from CT_datasets import load_data
from model_process import model_def, pre_process
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import datetime
import os
import matplotlib.pyplot as plt 
import numpy as np
###########################################################################
#define hyper-parameters
print('Defining hyper-parameters...')
lr1=0.001
decay1 = 5*1e-4
momentum1=0.9
epochs = 5
batch_size = 64
#define path where to store trained weights; change this path to desired path
weight_path = '/home/trupti/Bharadwaj Kss/results/'
if os.path.exists(weight_path)==False:
    os.makedirs(weight_path)
nwt = datetime.datetime.now #to get current datetime
img_rows, img_cols = 128, 128 #image size
###########################################################################
#Load train, val and test data and labels from Ct scan dataset
print('Loading dataset.....')
train_data, train_label, val_data, val_label, test_data, test_label = load_data()
###########################################################################
#preprocess the dataset
print('Pre-processing the data...')
data_mean = np.mean(train_data) #find mean of data
data_dev = np.std(train_data)  #find std dev of data
train_data = pre_process(train_data,data_mean,data_dev)
val_data = pre_process(val_data,data_mean,data_dev)
test_data = pre_process(test_data,data_mean,data_dev)
###########################################################################
#define the network:UNET, optimizer:SGD, loss function:binary cross entropy, performance metric: accuracy
print('Defining the architecture of network...')
model = model_def(img_rows, img_cols)
sgd = SGD(lr=lr1, momentum=momentum1, decay=decay1, nesterov=True) #optimizer
model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpointer = ModelCheckpoint(filepath=weight_path+'weights_best.hdf5', verbose=1, save_best_only=True) #save only best weights
###########################################################################
#train the network
print('Fitting/training the model....')
start_time = nwt() #start timer
result = model.fit(train_data, train_label,
          batch_size=batch_size,epochs=epochs,verbose=2,
          validation_data=(val_data, val_label),callbacks=[checkpointer]) #train network
end_time = nwt() #stop timer
model.save_weights(weight_path+'model_weights.hdf5',overwrite=True) #save weights
print('Training time: %s' % (end_time - start_time))
###########################################################################
#evaluate/test the data on trained model
print('Evaluating the model on testing set.....')
score = model.evaluate(test_data, test_label, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
###########################################################################
#function to plot the epochwise accuracy and loss of training and validation
# summarize history for accuracy
plt.figure()
plt.title('results in terms of accuracy')
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(weight_path+'accuracy plot')
plt.show()
# summarize history for loss
plt.figure()
plt.title('results in terms of loss')
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(weight_path+'loss plot')
plt.show()
###########################################################################
