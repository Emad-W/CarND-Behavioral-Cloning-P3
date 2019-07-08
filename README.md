# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository is based upon the Behavioral Cloning Project for the Self-Driving CAR NanoDegree by Udacity.

In this project, We utilized techniques from deep neural networks and convolutional neural networks to clone driving behavior using a simulator to simulate our environment and acquisation of our data set. Model was built using Keras for trainning, validation and testing purposes.
The model will output a steering angle to have control over an autonomous vehicle through the simulator.[Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)


**Project description:**
---
The steps of this project were as following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./Data/NvidiaArchitecture.png "Model Visualization"
[image2]: ./Data/capture1.png "Sample output Image"
[image3]: ./Data/Errorloss.png "mean square error"
[image4]: ./Data/capture2.png "Sharp turn Image"


---
### Files description

### My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Readme.md or writeup_report.pdf summarizing the results
* VideoIllustration.mp4 to preview samle of the result and model performance.

#### Drive.py
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the following command and then running the simulator.
```sh
python drive.py model.h5
```

#### Model.h5
This file contains the Keras Model weights as HDF5 by adding this line to the code
```model.save_weights('model.h5')```
To load this model weight later on, one can use ``` model.load_weights('model.h5') ```
 to simulate or inference the same model back again.



### Model Architecture and Training Strategy


The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



#### 1. An appropriate model architecture has been employed
My Model network architecture was based upon the [Nvidia model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) of its demonstrated suitability for self-driving car tasks. The architecture uses the images with a shape of 
(160, 320, 3) then Cropped and normalizedusing a Keras lambda layer (model.py line 77).


Afterwards it consists of convolution layers  with 5x5 and 3x3 filter sizes and depths between 24 and 64 (code lines 81-85) 

The model includes activation through RELU layers to introduce nonlinearity for each convolution operation, 
The model contains dropout was added to reduce overfitting of data(code line 86).

At last a fully connected layers of 100 then 50 then 10 neurons was introduced.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

Training data was also chosen with the mindset to ensure that the car has the most scenarios that can be face to keep the vehicle driving inside the road no matter what happened during simulation and what scenarios it has faced. 

A combination of center lane driving, recovering from the left and right sides of the road, Recovering from sharp turns and on the edge to teach the car what it should do in these kind of situations. 





### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to collect data from the 3 cameras existing on car, Add them to the dataset with correct consideration for each of the camera views and then shuffling the data set.

Also I split my data set into a training and validation set in order to validate the model and check for mean square error.

My first step was to use the model with only center images without considering corner case, this led to having the car drift away, not figuring out how to recover back on track.

Then I Included some specefic scenarios for car recovery from tough turns and some corner cases, Also the data from the 2 side cameras was included with steering angle correction factor to introduce more stability and robustness.
The final step was to run the simulator to see how well the car was driving around track one more time and the vehicle is able to drive autonomously around the track without leaving the road as show in VideoIllustration.mp4 file.

![alt text][image2]

Also when taking a sharp Edge the car stayed in the middle of the road as shown
![alt text][image4]
Also our calculated Mean square error 0.008 shows very good performance for our model afterwards,

![alt text][image3]

