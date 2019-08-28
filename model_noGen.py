import csv
import cv2
import numpy as np
import math

#The point of this part of the script is to change the paths in my data so that it refers to the
# data in the workspace instead of in my local computer. 
# Note: to get the data to my workspace, I zip it, upload it, and then unzip using the terminal
dataFolder = 'simData'
dataIMG = 'simData/IMG'
lines = []
with open('/opt/carnd_p3/' + dataFolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] != 'center': #firstline is the 
            lines.append(line) #path to image data
images = []
measurements = []
for line in lines:
    #Only look at data if the car is moving forward
    speed = float(line[6])
    if speed>1:
        for i in range(3):
            source_path = line[i]                    #used to be so                     
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
            measurement = (float(line[3]) + correction) #correct for left and right camera 
            measurements.append(measurement) #Extract steering angles from the data 
        


X_train = np.array(images)
y_train = np.array(measurements) 
        
    
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3) #epoch is 10 by defualt


#Now save the model so I can try it on my local machine
model.save('model2.h5')

