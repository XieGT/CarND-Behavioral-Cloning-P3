# **Behavioral Cloning Project** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model introduced by NVDIA autonomous driving team is adopted here. The structure are shown as follow:

1. the images are normalized and centralized using Keras lambda function.
```
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
```
2. the images are cropped using Keras Cropping2D 
```
model.add(Cropping2D(cropping=((70,25), (0,0))))
```
3. 5 convolutional layers, first three layers with kernel size 5x5, strides size 2x2; other two layers with kernel size 3x3.

 ```
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
 ```

4. Keras flatten function is used to connect the convolutional layers with fully connected layers. 
 
 ```
model.add(Flatten())
 ```
5. 3 fully connected layers introduced 
 
 ```
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
```
6. the output layer

 ```
model.add(Dense(1))
 ```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting

```
model.add(Dropout(0.25))
``` 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 85). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

```
model.compile(loss='mse',optimizer = 'adam')
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with a correction factor 0.2. Meanwhile, the original images are flipped to augment the data set.

```
augmented_images.append(image)
augmented_measurements.append(measurement)
augmented_images.append(cv2.flip(image,1))
augmented_measurements.append(measurement*-1.0)
```

