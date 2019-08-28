import csv
import cv2
import numpy as np
import math

#The point of this part of the script is to change the paths in my data so that it refers to the
# data in the workspace instead of in my local computer. 
# Note: to get the data to my workspace, I zip it, upload it, and then unzip using the terminal

lines = []
#dataFolder = 'data' #built in data
#dataIMG = 'data'
#dataFolder = 'AaronData' #data I collected
#dataIMG = 'AaronData/IMG'
dataFolder = 'simData'
dataIMG = 'simData/IMG'
with open('/opt/carnd_p3/' + dataFolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] != 'center': #firstline is the 
            lines.append(line) #path to image data
#print("Lines: "+ str(len(lines)))
#Use generator to load data and preprocess it in batch size portions
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
import sklearn

def generator(samples, batch_size=64, train = True):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            center_images = []
            center_measurements = []
            #train = False #Lets only look at center image
            if train==True:
                numImages = 3
            else:
                numImages = 1
            for batch_sample in batch_samples:
                #print("batch_sample: "+ str(batch_sample))
                for i in range(numImages):
                    source_path = batch_sample[i]                    #used to be so                     
                    filename = source_path.split('\\')[-1]
                    filename = filename.split('/')[-1] #this may mess up my previous data files
                    current_path = '/opt/carnd_p3/' + dataIMG + '/' + filename
                    #print("current_path: " + str(current_path))
                    image =cv2.imread(current_path)
                    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(rgb_img) #Extract images from the data
                    correction = 0.0 #For center image
                    if i==1: #left
                        correction = 0.3 #For left camera 
                    elif i==2: #right 
                        correction = -0.3 #For right camera
                    measurement = (float(batch_sample[3]) + correction) #correct for left and right camera 
                    measurements.append(measurement) #Extract steering angles from the data 
                    #Now if the image is a center image store it in a separate list so we can flip it later for image augmentation
                    if i==0: #center image
                        center_images.append(image)
                        center_measurements.append(measurement)
                 
            #Now augment the images by mirroring them (nd inverting steering angles) so the car isnt biassed to turning left
            augmentImages = True
            if augmentImages==True and train==True:
                for image,measurement in zip(center_images, center_measurements):
                    if measurement > 0: #only flip if there was steering
                        images.append(cv2.flip(image,1))
                        measurements.append(measurement*-1.0)  
         
            X_train = np.array(images)
            y_train = np.array(measurements) 
            yield sklearn.utils.shuffle(X_train, y_train)
        
#Set our batch size
batchSize = 128
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batchSize, train = True)
validation_generator = generator(validation_samples, batch_size=batchSize, train = False)
    
#Now set up the regression network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

model = Sequential()

#Normalize and mean center the model
model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0)))) 
model.add(Convolution2D(24,5,5,activation="relu", subsample=(2,2), name='conv_1'))
model.add(Convolution2D(36,5,5,activation="relu", subsample=(2,2), name='conv_2'))
model.add(Convolution2D(48,5,5,activation="relu", subsample=(2,2), name='conv_3'))
model.add(Convolution2D(64,3,3,activation="relu", name='conv_4'))
model.add(Convolution2D(64,3,3,activation="relu", name='conv_5'))
model.add(Dropout(0.5)) #Added to reduce overfitting
model.add(Flatten(name="flat_1"))
model.add(Dense(100,name="dense_1"))
#model.add(Dropout(0.5)) #Added to reduce overfitting
model.add(Dense(50,name="dense_2"))
#model.add(Dropout(0.5)) #Added to reduce overfitting
model.add(Dense(10,name="dense_3"))
#model.add(Dropout(0.5)) #Added to reduce overfitting
model.add(Dense(1,name="dense_4"))

model.compile(loss='mse', optimizer='adam')
model.summary()

model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)*3/batchSize), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batchSize), epochs=3, verbose=1) #epoch is 10 by defualt



#Now save the model so I can try it on my local machine
model.save('model.h5')


