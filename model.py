from scipy import ndimage
import csv
import cv2
import numpy as np
import os
import math

samples = []
path = '../data/Behavioral_Clonning/'
csv_file = path + 'driving_log.csv'
with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                path_IMG = 'IMG/' # fill in the path to your training IMG directory
                img_center = np.asarray(ndimage.imread(path + path_IMG + batch_sample[0].split('\\')[-1]))
                img_left = (np.asarray(ndimage.imread(path + path_IMG + batch_sample[1].split('\\')[-1])))
                img_right = (np.asarray(ndimage.imread(path + path_IMG + batch_sample[2].split('\\')[-1])))
                    
                # add images and angles to data set
                images.append(img_center)
                angles.append(steering_center)
                images.append(img_left)
                angles.append(steering_left)
                images.append(img_right)
                angles.append(steering_right)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 80, 320  # Trimmed image format

# Setup Keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()

model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape = (160,320,3)))

model.add(Cropping2D(cropping=((60,20),(0,0))))

model.add(Conv2D(24,(5,5), strides = (2,2), activation = "relu"))
model.add(Conv2D(36,(5,5), strides = (2,2), activation = "relu"))
model.add(Conv2D(48,(5,5), strides = (2,2), activation = "relu"))
model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(Dropout(0.5))
model.add(Flatten())   
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))





model.compile(loss ='mse', optimizer = 'adam')
#model.fit(x_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

#from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=8, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('History_object-4.png')

model.save('model-4.h5')
exit()







