# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:43:48 2019

@author: trupti
"""

#####################################################
#Aim: Loading/Reading data and generate .npy files of data for training of network
#Train data contains CT scans of 7 patients
#Test data contains CT scans of 3 patients
#validation data is randomly chosen from 30% of training data
#Note: The images are resized to 128*128 as compromise in training time and processing. 
#To process 512*512 image, variables img_rows,img_cols are needed to set to 512
#1. Data Cleaning
#From dataset it is observed that, few images are redudant since they does not contain any data (completely black).
#Such images are discarded from the dataset 
#2.Minority labelled class problem
#There are very few samples having mask/abnormality ground truth available in dataset.
#Hence, such data is oversampled to have balance in both the normal and abnormality classes.
#Data augmentation is done by rotation, translation and scaling
#3.Data is formatted as per the requirement of network input and output
#input is 4D numpy array and output is 0/1 labels.
#inputs are accordingly reshaped and output 0-255 range is converted to 0-1.
#####################################################
#import necessary packages
from __future__ import print_function
import sys, os
import glob
from skimage.io import imread
#from skimage import img_as_ubyte, exposure
from skimage.transform import AffineTransform, warp,resize, rotate, rescale
#from random import shuffle
import numpy as np
import cv2
#import math
#####################################################
#define variables, constants
img_rows, img_cols = 128, 128
#path to find the dataset; Change this path to the folder containing the dataset files
data_path = '/home/trupti/Bharadwaj Kss/sample_dataset_for_testing/sample_dataset_for_testing/fullsampledata/'
subnet_list = sorted(os.listdir(data_path)) #list of folders subnetXmask
#shuffle(subnet_list)
#####################################################
#function to check whether image is redundant or not; Completely blck images are discarded
#input: image
#ouput: boolean flag
def remove_blackimg(img):    
    if np.all(img==0): #check: if all pixels in image are black then return true indication else return false indication.
        return True
    else:
        return False
#####################################################
#function to roate images by angles in range -5 to 5 for data augmentation
#input: img and labeled mask image
#output: set of rotated image and mask image
def apply_rotation(img, mask):
    rot_imgs = []
    rot_masks = []
    angles = np.radians(np.arange(-5,5,0.2)) #angles range: -5:0.2:5
    for i in range(len(angles)):
        tf = AffineTransform(rotation=angles[i])
        rot_imgs.append(warp(img, tf)) #rotate image and append rotated image to a list
        rot_masks.append(warp(mask, tf)) #rotate mask image and append rotated mask image to a list
#        rot_imgs.append(rotate(img, angles[i], resize = False)) #rotate image and append rotated image to a list
#        rot_masks.append(rotate(mask, angles[i], resize = False)) #rotate mask image and append rotated mask image to a list
    return rot_imgs, rot_masks
#####################################################
#function to translate images in range -5 to 5 for data augmentation
#input: img and labeled mask image
#output: set of translated image and mask image
def apply_translations(img, mask):
    trans_imgs = []
    trans_masks = []
    shifts = np.arange(-5,5,1) #shifting range: -5:1:5
    for i in range(len(shifts)):
        tf = AffineTransform(translation=(shifts[i],shifts[i]))
        trans_imgs.append(warp(img,tf)) #translate image and append translated image to a list
        trans_masks.append(warp(mask,tf)) #translate mask image and append translated mask image to a list
    return trans_imgs, trans_masks        
#####################################################
#function to scale images in range 0.7 to 1.2 for data augmentation
#input: img and labeled mask image
#output: set of scaled image and mask image
def apply_scaling(img, mask):
    scale_imgs = []
    scale_masks = []
    scales = np.arange(0.7,1.2,0.1) #scaling range: 0.7:0.1:1.2
    for i in range(len(scales)):
        tf = AffineTransform(scale=(scales[i],scales[i]))
        scale_imgs.append(warp(img, tf)) #scale image and append scaled image to a list
        scale_masks.append(warp(mask, tf)) #scale mask image and append scaled mask image to a list        
    return scale_imgs, scale_masks
#####################################################
#function to oversample class with minority labeled images from the dataset
#input: pat_masklist:list of patients' mask
#output: set of augmented image and mask image
def oversample(pat_masklist):
    aug_imgs = []
    aug_masks = []
    for i in range(len(pat_masklist)):
        img = imread(pat_masklist[i][:-10]+'.tiff', as_grey=True).astype('uint8') # read the image
        img = resize(img,output_shape = (img_rows,img_cols))
        #img = (exposure.rescale_intensity(img, out_range=(0, 255))).astype('uint8')#2**31 - 1)))
        mask = imread(pat_masklist[i], as_grey=True).astype('uint8') #read the masked image
        mask = resize(mask,output_shape = (img_rows,img_cols))
        rot_imgs, rot_masks = apply_rotation(img, mask) #apply rotation transform
        trans_imgs, trans_masks = apply_translations(img,mask) #apply translation transform
        scale_imgs, scale_masks = apply_scaling(img,mask) #apply scaling transform
        aug_imgs = aug_imgs+rot_imgs+trans_imgs+scale_imgs #concatenated all transformed images
        aug_masks = aug_masks+rot_masks+trans_masks+scale_masks #concatenated all transformed mask images       
    return aug_imgs,aug_masks
#####################################################
#function to convert 0-255 label to 0/1 label
def conv_to_binlabel(label):
    bin_label=label>0
    bin_label=bin_label.astype(np.int)
    return bin_label
#####################################################
#function to oversample class with minority labeled images from the dataset to generate train data
#input: subnet_start: start index of subnetXmask,subnet_end: end index of subnetXmask,
#nb_samples: number of samples to be selected from normal and abnormal classes, param: name of set, train/val
#output: set of train, val data and respective labels 
def read_train_data(subnet_start,subnet_end,nb_samples, param):
    print('Preparing ',param,'set......')
#subnet_start,subnet_end = 0,4
    norm_data = [] #list of data for normal class
    norm_label = [] #list of labels for normal class
    osamp_data = [] #list of oversampled data for abnormal class
    osamp_label = [] #list of oversampled labels for abnormal class
#    nb_samples = 2100 #number of samples from normal and abnormal classes to be chosen from train set
    for i in range(subnet_start,subnet_end):  #do for all folders in complete dataset from subnet_start to subnet_end
        pat_rec_path = data_path+subnet_list[i]+'/'  #path of subnetXmask folder
        pat_reclist = sorted(os.listdir(pat_rec_path)) #list of patients in subnetXmask folder
        for j in range(len(pat_reclist)): #do for all patients in subnetXmask folder
            pat_img_path = pat_rec_path+pat_reclist[j]+'/' #path for CT scan images for jth patient in subnetXmask folder
            pat_imglist = sorted(glob.glob(pat_img_path+'*.tiff')) #list of all CT scan images for jth patient in subnetXmask folder
            pat_masklist = sorted(glob.glob(pat_img_path+'*mask.tiff')) #list of only masked/labeled images for jth patient in subnetXmask folder        
            for k in range(len(pat_imglist)): #do for all images for jth patient in subnetXmask folder
                #find normal images using the name of images; images having 'mask.tiff' are abnormal, while without 'mask.tiff' are normal
                if ((pat_imglist[k] not in pat_masklist)&(pat_imglist[k][:-5]+'_mask.tiff' not in pat_masklist)):  #if normal
                    img = imread(pat_imglist[k], as_grey=True).astype('uint8') #read the image
                    img = resize(img,output_shape = (img_rows,img_cols)) 
                    if remove_blackimg(img)==False: #check if it is redundant; if all pixels in image are black then it it redundant
                        norm_data.append(img) #if image not redandant then append it to list of normal class
                        norm_label.append(np.zeros((img_rows,img_cols))) #append its label to list of normal class
            osamp_img, osamp_masks = oversample(pat_masklist) #oversample all abnormal images for jth patient in subnetXmask folder
            osamp_data = osamp_data + osamp_img  #concatenate oversampled images for all patients in subnetXmask folder and all subnetXmask folder
            osamp_label = osamp_label + osamp_masks  #concatenate oversampled mask imsges for all patients in subnetXmask folder and all subnetXmask folder
    ###############shuffle the normal and augmented abnormal class data#####################################
    ########Abnormal data#########
    #convert list to array
    osamp_data = np.array(osamp_data)
    osamp_label = np.array(osamp_label)
    #index array for shuffling
    shuffle_id = np.arange(len(osamp_data))
    #shuffle indices
    np.random.shuffle(shuffle_id) 
    #shuffle data according to shuffled indices
    osamp_data = osamp_data[shuffle_id]
    osamp_label = osamp_label[shuffle_id]         
    
    ########Normal data#########
    #convert list to array
    norm_data = np.array(norm_data)
    norm_label = np.array(norm_label)
    #index array for shuffling
    shuffle_id = np.arange(len(norm_data))
    #shuffle indices
    np.random.shuffle(shuffle_id)
    #shuffle data according to shuffled indices
    norm_data = norm_data[shuffle_id]
    norm_label = norm_label[shuffle_id]    
    ##############Choose nb_samples from normal and abnormal data#############
    #This is required step to have balanced normal and abnormalities data in training and validation set   
    #divide data into training (70%) and validation (30%) set
    nb_train_samples = int(nb_samples*0.7)  #number of training samples(70%)
    norm_train_data = norm_data[0:nb_train_samples]
    norm_train_label = norm_label[0:nb_train_samples]
    osamp_train_data = osamp_data[0:nb_train_samples]
    osamp_train_label = osamp_label[0:nb_train_samples]
    
    norm_val_data = norm_data[nb_train_samples:nb_samples]
    norm_val_label = norm_label[nb_train_samples:nb_samples]
    osamp_val_data = osamp_data[nb_train_samples:nb_samples]
    osamp_val_label = osamp_label[nb_train_samples:nb_samples]
    ##########combine normal and oversampled abnormal data##############
    train_data = np.zeros((norm_train_data.shape[0]+osamp_train_data.shape[0],norm_train_data.shape[1],norm_train_data.shape[2]))
    train_label = np.zeros((norm_train_data.shape[0]+osamp_train_data.shape[0],norm_train_data.shape[1],norm_train_data.shape[2]))
    train_data[0:norm_train_data.shape[0],:,:] = norm_train_data
    train_data[norm_train_data.shape[0]:,:,:] = osamp_train_data
    train_label[0:norm_train_data.shape[0],:,:] = norm_train_label
    train_label[norm_train_data.shape[0]:,:,:] = osamp_train_label
    train_data = train_data.reshape((train_data.shape[0],train_data.shape[1],train_data.shape[2],1)) #reshape to 4-d np array
    
    val_data = np.zeros((norm_val_data.shape[0]+osamp_val_data.shape[0],norm_val_data.shape[1],norm_val_data.shape[2]))
    val_label = np.zeros((norm_val_data.shape[0]+osamp_val_data.shape[0],norm_val_data.shape[1],norm_val_data.shape[2]))
    val_data[0:norm_val_data.shape[0],:,:] = norm_val_data
    val_data[norm_val_data.shape[0]:,:,:] = osamp_val_data
    val_label[0:norm_val_data.shape[0],:,:] = norm_val_label
    val_label[norm_val_data.shape[0]:,:,:] = osamp_val_label
    val_data = val_data.reshape((val_data.shape[0],val_data.shape[1],val_data.shape[2],1)) #reshape to 4-d np array
    #############shuffle normal and abnormal data#############
    #index array for shuffling
    shuffle_id = np.arange(len(train_data))
    #shuffle indices
    np.random.shuffle(shuffle_id) 
    #shuffle data according to shuffled indices
    train_data = train_data[shuffle_id]
    train_label = train_label[shuffle_id] 
    train_label = train_label.reshape((train_label.shape[0],train_label.shape[1],train_label.shape[2],1)) #reshape to 4-d np array
    
    #index array for shuffling
    shuffle_id = np.arange(len(val_data))
    #shuffle indices
    np.random.shuffle(shuffle_id) 
    #shuffle data according to shuffled indices
    val_data = val_data[shuffle_id]
    val_label = val_label[shuffle_id] 
    ############convert label to 0/1 labels#########
    train_label = conv_to_binlabel(train_label)
    val_label = conv_to_binlabel(val_label)
    val_label = val_label.reshape((val_label.shape[0],val_label.shape[1],val_label.shape[2],1)) #reshape to 4-d np array
    return train_data, train_label, val_data, val_label
#####################################################
#function to oversample class with minority labeled images from the dataset to generate test data
#input: subnet_start: start index of subnetXmask,subnet_end: end index of subnetXmask,
#param: name of set, train/val
#output: set of test data and labels 
def read_test_data(subnet_start,subnet_end,param):
    print('Preparing ',param,'set......')
#    subnet_start,subnet_end = 7,10
    data = [] #list of data for normal class
    label = [] #list of labels for normal class
    for i in range(subnet_start,subnet_end):  #do for all folders in complete dataset from subnet_start to subnet_end
        pat_rec_path = data_path+subnet_list[i]+'/'  #path of subnetXmask folder
        pat_reclist = sorted(os.listdir(pat_rec_path)) #list of patients in subnetXmask folder
        for j in range(len(pat_reclist)): #do for all patients in subnetXmask folder
            pat_img_path = pat_rec_path+pat_reclist[j]+'/' #path for CT scan images for jth patient in subnetXmask folder
            pat_imglist = sorted(glob.glob(pat_img_path+'*.tiff')) #list of all CT scan images for jth patient in subnetXmask folder
            pat_masklist = sorted(glob.glob(pat_img_path+'*mask.tiff')) #list of only masked/labeled images for jth patient in subnetXmask folder   
            processed_flag = []
            for k in range(len(pat_imglist)): #do for all images for jth patient in subnetXmask folder
                #find normal images using the name of images; images having 'mask.tiff' are abnormal, while without 'mask.tiff' are normal
                if ((pat_imglist[k] not in pat_masklist)&(pat_imglist[k][:-5]+'_mask.tiff' not in pat_masklist)):  #if normal
                    img = imread(pat_imglist[k], as_grey=True).astype('uint8') #read the image
                    img = resize(img, output_shape = (img_rows,img_cols))
                    if remove_blackimg(img)==False: #check if it is redundant; if all pixels in image are black then it it redundant
                        data.append(img) #if image not redandant then append it to list of normal class
                        label.append(np.zeros((img_rows,img_cols))) #append its label to list of normal class
                else:
                    if ((pat_imglist[k] not in processed_flag)&(pat_imglist[k][:-5]+'_mask.tiff' not in processed_flag)): #if any of image or mask image already considered, skip other
                        img = imread(pat_imglist[k], as_grey=True).astype('uint8') #read the image
                        img = resize(img, output_shape = (img_rows,img_cols))
                        data.append(img) #append it to list
                        mask = imread(pat_imglist[k][:-5]+'_mask.tiff', as_grey=True).astype('uint8')
                        label.append(resize(mask, output_shape = (img_rows,img_cols))) #append its label to list of normal class
                        processed_flag.append(pat_imglist[k])
                        processed_flag.append(pat_imglist[k][:-5]+'_mask.tiff')
    ###############shuffle the normal and augmented abnormal class data#####################################
    #convert list to array
    data = np.array(data)
    label = np.array(label)
    #index array for shuffling
    shuffle_id = np.arange(len(data))
    #shuffle indices
    np.random.shuffle(shuffle_id)
    #shuffle data according to shuffled indices
    data = data[shuffle_id,:,:]
    label = label[shuffle_id,:,:]    
    data = data.reshape((data.shape[0],data.shape[1],data.shape[2],1)) #reshape to 4-d np array
    label = conv_to_binlabel(label) #convert label to binary
    label = label.reshape((label.shape[0],label.shape[1],label.shape[2],1)) #reshape to 4-d np array
    return data, label
#####################################################
#Function for loading the dataset
#Steps:
#1. Read the training data and labels
#2. Divide the set into 70% training and 30% testing set.
#read the test data and labels
def load_data():
    train_data, train_label, val_data, val_label = read_train_data(0,7,500, 'train')   #read the training data and labels    
    #####read test data####
    #no need to oversample the abnormal data
    test_data, test_label = read_test_data(7,10,'test')
    return train_data, train_label, val_data, val_label, test_data, test_label
#####################################################
