#############################################################################
Problem Statement:
To design a Convolutional Neural Network to identify abnormalities in 3D Chest CT scans. 
#############################################################################
Solution:
The solution contains three files and 1 folder:
Files:
1. CT_datasets.py
2. model_process.py
3. train_eval.py
Folder: results: It contains the weight files and plot of epochwise training.

The solution of problem statement is given in file 'train_eval.py'. 
To run the solution file:
Change the path of dataset in file CT_datasets.py as per requirement. (data_path = '....')
Also change the path to which weights are to be saved in train_eval.py. (weight_path = '....')
The details of these files are discussed as follows:
#############################################################################
1. CT_datasets.py: includes functions required to load the dataset and generate .npy files of data for training of network
Train data contains CT scans of 7 patients
Test data contains CT scans of 3 patients
validation data is randomly chosen from 30% of training data
1. Data Cleaning
From dataset it is observed that, few images are redudant since they does not contain any data (completely black).
Such images are discarded from the dataset 
2.Minority labelled class problem
There are very few samples having mask/abnormality ground truth available in dataset.
Hence, such data is oversampled to have balance in both the normal and abnormality classes.
Data augmentation is done by rotation, translation and scaling.
3.Data is formatted as per the requirement of network input and output
input is 4D numpy array and output is 0/1 labels.
inputs are accordingly reshaped and output 0-255 range is converted to 0-1.
Notes: 
1. The above steps can also done using data visualization, cleaning tools such as tableu, orange, available data augmentation techniques, SMOTE, etc.
2. The images are resized to 128*128 as compromise in training time and processing. 
   To process 512*512 image, variables img_rows,img_cols are needed to set to 512
3. Since the code was executed on CPU, the training samples are reduced to 2000 (train:1400, val:600). This can be modified to cover whole dataset.
#############################################################################
2. model_process.py: includes function to define the network architecture and preprocessing.
Unet is considered as network architecture. The architecture is modified to reduce training time.
For pre-processing, data is normalized using mean and standard deviation.
#############################################################################
3. train_eval.py: This is the main to be run
It is used to train the CNN for segmentation of abnormality.
Training details: SGD is used for optimization.
learning rate=0.001, weight decay = 5*1e-4, momentum=0.9, batch size = 64.
Training is carried for 5 epochs. This can be increased for improving the accuracy.
The steps carried for training are:
1. Load the train, val and test data.
2. Define the architecture of model.
3. Train the network.
4. Evaluate the network.
#############################################################################

Results:
Defining hyper-parameters...
Loading dataset.....
Preparing  train set......
Preparing  test set......
Pre-processing the data...
Defining the architecture of network...
Fitting/training the model....
Train on 700 samples, validate on 300 samples
Epoch 1/5
 - 22s - loss: 0.6673 - acc: 0.9995 - val_loss: 0.6521 - val_acc: 0.9995

Epoch 00001: val_loss improved from inf to 0.65209, saving model to /home/trupti/Bharadwaj Kss/results/weights_best.hdf5
Epoch 2/5
 - 22s - loss: 0.6323 - acc: 0.9995 - val_loss: 0.6048 - val_acc: 0.9995

Epoch 00002: val_loss improved from 0.65209 to 0.60480, saving model to /home/trupti/Bharadwaj Kss/results/weights_best.hdf5
Epoch 3/5
 - 22s - loss: 0.5765 - acc: 0.9995 - val_loss: 0.5391 - val_acc: 0.9995

Epoch 00003: val_loss improved from 0.60480 to 0.53909, saving model to /home/trupti/Bharadwaj Kss/results/weights_best.hdf5
Epoch 4/5
 - 22s - loss: 0.5010 - acc: 0.9995 - val_loss: 0.4494 - val_acc: 0.9995

Epoch 00004: val_loss improved from 0.53909 to 0.44941, saving model to /home/trupti/Bharadwaj Kss/results/weights_best.hdf5
Epoch 5/5
 - 22s - loss: 0.3924 - acc: 0.9995 - val_loss: 0.3128 - val_acc: 0.9995

Epoch 00005: val_loss improved from 0.44941 to 0.31281, saving model to /home/trupti/Bharadwaj Kss/results/weights_best.hdf5
Training time: 0:01:58.525436
Evaluating the model on testing set.....
1004/1004 [==============================] - 8s 8ms/step
('Test loss:', 0.28508640154899356)
('Test accuracy:', 0.9999896045700013)



