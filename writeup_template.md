# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/network_visualization.svg "Model Visualization"
[image2]: ./examples/center_lane_driving.jpg "Center lane driving"
[image3]: ./examples/recover_1.jpg "Recovery Image"
[image4]: ./examples/recover_2.jpg "Recovery Image"
[image5]: ./examples/recover_3.jpg "Recovery Image"
[image6]: ./examples/image.jpg "Normal Image"
[image7]: ./examples/image_flipped.jpg "Flipped Image"
[image8]: ./examples/image_cropped.jpg "Cropped image"
[image9]: ./examples/validation_loss_ep5.png "Validation loss after 5 epochs"
[image10]: ./examples/validation_loss_ep10.png	"Validation loss after 15 epochs"

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

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 36 (model.py lines 119-123) followed by 3 fully connected layers (model.py lines 125-127).

RELU activation is used in order to introduce non-linearities with every convolution. 

To normalize the data I used a Keras Lambda Layer. (model.py line 118)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 121). 

Training and validation data was created using a generator. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 129).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build on the model introduced in the lesson. But with getting totally different results on the dataset provided by Udacity, I figured that this approach would not get me very far.

After that I wanted to try the InceptionV3 network provided by Keras. This time I ran into a problem running the model in GPU mode which could be traced back to an outdated version of Tensorflow in the Udacity workspace. This issue also happened with every network provided with Keras. So I decided to build a network myself.

So I startet with a very reduced implementation of the NVIDIA network mentioned in the lesson.

I started with one convolution and two fully connected layers, which already gave me a good performance on the dataset provided by Udacity.

I choose to split the training data and validation data 80 percent to 20 percent.

Using the created model on the simulator achieved a stable steering in the straight parts of the track but having some issues in the curved parts.

So I decided to add another convolution and another fully connected layer as the model seems to underfit. 
Getting better performance on the improved model I started to look at the training data which was very unbalanced. So I decided to record my own data for training and validation.

In one of the next chapters I will describe in detail what I did to get better training data.

In order to prevent overfitting and  to get a more stable model I introduced a dropout layer with a dropout rate of 0.5 after the first convolution.

The resulting network managed to drive the first track with acceptable performance and managed all curves pretty well. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 116 - 129) consisted of a convolution neural network with the following layers and layer sizes:


|Layer (type)      |           Output Shape      |        Param # |
|:-----------------|:----------------------------|:---------------|
|cropping2d_1 (Cropping2D) |   (None, 80, 320, 3)    |    0       |
|lambda_1 (Lambda)         |   (None, 80, 320, 3)    |    0       |
|conv2d_1 (Conv2D)         |   (None, 38, 158, 24)   |    1824    |
|dropout_1 (Dropout)       |   (None, 38, 158, 24)   |    0       |
|conv2d_2 (Conv2D)         |   (None, 17, 77, 36)    |    21636   |
|flatten_1 (Flatten)       |   (None, 47124)         |    0       |
|dense_1 (Dense)           |   (None, 120)           |    5655000 |
|dense_2 (Dense)           |   (None, 10)            |    1210    |
|dense_3 (Dense)           |   (None, 1)             |    11      |

Total params: 5,679,681
Trainable params: 5,679,681
Non-trainable params: 0

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. And after that I recorded three laps of center lane driving in the opposite direction. 
Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center.
These images show what a recovery looks like starting from the right:

![alt text][image3]
![alt text][image4]
![alt text][image5]



To augment the data sat, I also flipped images and angles thinking that this would balance the data even more.
For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

In addition to the center and flipped image, I also used the left and right images with a correction factor of 0,3 degree to train on.

I also cropped the images so the model does not train on surroundings, but rather on the road itself.
![cropped_img][image8]

Then I did two rounds on the second track to get a more generalized model.

After the collection process, I had 13555 number of data points. With all the augmented data I had 54220 data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

I first trained the model for 5 epochs, which was sufficient to drive the track without leaving the road. To see if I could improve the driving behavior even more, I trained the model for another 10 epochs. With the validation and the training loss dropping I assumed that the driving behavior would improve, but at least no improvement could be seen. So I decided not to train the model any further.

These are the losses after 5 epochs:

![][image9]

These are the losses after training the model for another 10 epochs:

![][image10]

### Discussion

For the first track the performance is pretty good and the car stays on the road. In order to improve the behavior the model, more training data should be recorded. 
Also the model can be trained for more epochs, but it seems much more epochs are needed. 
For driving on the second track I might try to add another convolutional layer and see if that improves performance.