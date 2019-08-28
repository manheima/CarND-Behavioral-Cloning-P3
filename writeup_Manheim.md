# **Behavioral Cloning** 

## Writeup

Aaron Manheim

7/8/2018

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./report_pics/nvidia_architecture.png "NVIDIA Model"
[image1]: ./report_pics/model.png "Model"
[image2]: ./report_pics/center.jpg "Center"
[image3]: ./report_pics/left1.jpg "Left" 
[image4]: ./report_pics/left2.jpg "Left 2" 
[image5]: ./report_pics/left3.jpg "Left 3"
[image6]: ./report_pics/unflipped.jpg "Normal Image"
[image7]: ./report_pics/flipped.jpg "Flipped Image"

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
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used a model developed by the NVIDIA autonomous driving team. 
(https://devblogs.nvidia.com/deep-learning-self-driving-cars/) 

The network consits of a normalization layer foloowed by 5 convolutional layers and then 4 fully-connected layers. 

![NVIDUA Model][image0]

The network model can be found in lines 94-105 in model.py.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 25). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for at least one lap.

I also added a dropout layer after the convolutions in order to combat overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road. I also gathered data going the opposite direction in the track in order to avoid overfitting. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple to ensure that the data was being consumed correctly by the model and that the model was outputting appropriate steering angles. In the beginning I used a simple LeNet architecture model (models_simple.py lines 52-60). 

Once I saw that this model was working decently I switched to the NVIDIA mdoel which had much more layers and took longer to train. It resutled in a much better driving model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it took the mirror image of the images in which the steering angle was greater than 0. This helped fight against the left turn bias since the track had mostly left turns. 

I also augmented the data with the left and right camera images. I did this by adding a manual offset for the angle of 0.22 to the left and right images. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In those cases, I took more data around those points and then retrained the model. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 94-105) was based off of the NVIDIA model described earlier.

Here is a visualization of the architecture:

![model][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn what to do if it got off center. These images show what a recovery looks like starting from the left.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help reduce the bias of turning to the left since most of the turns in the track are left turns. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 9018 center images. Since I also used the left and right cameras, that trippled the amount of data points to 27054.  I then preprocessed this data by noramlizing the pixel values between -0.5 to 0.5 and cropping the top and bottom of the image. 


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the decreasing loss values which plateaued around 4 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
