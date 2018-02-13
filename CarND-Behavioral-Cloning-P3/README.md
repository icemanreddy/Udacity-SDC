# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center_2018_02_13_00_06_42_397 "image"
[image2]: ./center_2018_02_13_00_06_42_397_cropped.jpg "cropped"
[image3]: ./data0hist_after.png "Histogram after correcting for small steering angles"
[image4]: ./data2hist_after.png "Histogram after correcting for small steering angles"
[image5]: ./data0hist_before.png "Histogram of sample data"
[image6]: ./data2hist_before.png "Histogram of collected data""

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator (mac_sim) and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive_orig.py model.h5
```


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with below model.It is the nvidia model with 3 dropout layers.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 62, 196, 24)   1824        lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 58, 192, 36)   21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 54, 188, 48)   43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 54, 188, 48)   0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 52, 186, 64)   27712       dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 52, 186, 64)   0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 50, 184, 64)   36928       dropout_2[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 588800)        0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 588800)        0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           58880100    dropout_3[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 59,017,019
Trainable params: 59,017,019
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### 2. Attempts to reduce overfitting in the model

The model contains 3 dropout layers with different rates in order to reduce overfitting . I have set a
lower dropout rate in initial layers since they capture most of the important features and a higher rate after the flattening layer.


#### 3. Model parameter tuning

The model uses an adam optimizer with the learning rate set to 0.0001 (Default is 0.01) and decay rate set to 0.00001.
On each subsequent iteration, the learning rate is set to 0.1*old_lr

#### 4. Appropriate training data

The training data was the sample data and then data from two runs of the track by me.

![][image1]

![][image2]
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a LeNet base architecture and kept increasing the model size to try and improve the performance.
I iteratively added training data to the model but after one point it stopped learning .That was the previous model that I submitted.I had trained it to for 3-4 epochs only.It's performance was not satisfactory.

I was also working on the nvidia model on the model but was unable to get it to work.

I finally decided to recollect data and ensure that its high quality.I did this because the nvidia model is a proven model and the only variable is the data being collected.
After collecting fresh data, reducing the  instances of 0 steering angles for some iterations and adding dropouts ,the nvidia model works well for track1.

#### 2. Creation of the Training Set & Training Process

I have used the sample data provided and in addition also captured driving the tracks myself.
I didn't have to capture any recovery data.
I captured the data on the simulator that gave 3 images (left,right and center) .It was captured in 640x480 .But I dont think that has any affect on the collected image data.
I trained to 7 epochs for each training set of data .The validation and training loss was  about 0.03 for most of the iterations.

Each training data set was augmented by adding the flipped the image to the training data. I also use the left and right image and their flipped images to the training data.
I found a correction of 0.2 to work well


I have the data in 2 folders.one for sample data from udacity and the other that I collected.
Folder structure is as follows:

data/0/IMG

data/0/driving_log.csv

data/2/IMG

data/2/driving_log.csv


I have two options for training .One tries to evenly distribute the input images to reduce the number of 0 steering images.
The other method just trains on all the images.


![][image5]

![][image6]

![][image3]

![][image4]


##### Training process
I ran 1 iteration on the sample data till 7 epochs with the input images evenly distributed.
Followed by another iteration on top of  the last model with my data for 7 epochs with the input images evenly distributed.
I then ran an iteration of the original sample data with all images.

This model worked well on track1.



I was forced to use a batchsize of 4 because many a times I got errors related to memory shortage while training.


###### Issues faced

- The simulator gave me issues initially when the car would simply stop mid track or its speed would change haphazardly .I made a few changes to drive.py so that the throtle and speed work correctly.


- The simulation is not always the same and the model can do with some further fine tuning.
