import os
import csv
import cv2
import matplotlib.image as mpimg
import numpy as np 

# 1 Data Preparation

## 1.1 Read data from CVS file
lines = []
with open('./data/driving_log.cvs') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
del lines[0]

images, measurements = [], []

for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		image = mpimg.imread(current_path)
		images.append(image)

		measurement = float(line[3])
		# create adjusted steering measurements for the side camera images
		correction = 0.2
		if i == 1:
			measurement += correction
		elif i == 2:
			measurement -= correction

		measurements.append(measurement)

## 1.2 get the mirrored data
augmented_images, augmented_measurements = [],[]

for image,measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

## 1.3 define the training set 
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# 2 Build the NN Structured by NVIDIA

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Dropout 
from keras.layers.convolutional import Conv2D

model = Sequential()

## 2.1 normalize & centeralize the data 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

## 2.2 crop the image 
model.add(Cropping2D(cropping=((70,25), (0,0))))

## 2.3 define 5 Conv layers, first 3: 5x5, other 2: 3x3
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))

## 2.4 flatten the data and add another 3 fully connected layers
model.add(Flatten())
### to prevent overfitting enable dropout of 25%
model.add(Dropout(0.25))
model.add(Dense(100))
### to prevent overfitting enable dropout of 25%
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# 3. Train the NN 
model.compile(loss='mse',optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=8, batch_size=32)

# 4. Save the model
model.save('model.h5')






