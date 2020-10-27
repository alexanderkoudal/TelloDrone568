import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import cv2 as cv
import os
import tkinter as tk
import random
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import time


Name = 'Fruits-CNN-{}'.format(int(time.time()))


tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))


#global variables
Data_directory = "/home/manuela/Documents/Robotics4/mini project/Miniproject/Fruits" #variable of directory path
Categories = ['Apple', 'Banana', 'Cocos', 'GrapeBlue'] #list of directories inside FRUITS
    #THIS LOADS THE IMAGES INSIDE THE FOLDER AND PRINT THE IMAGES
#foor loop used to iterate through a sequence(list in this case)
for category in Categories: #create an object category for Categories - substitures the prev Categories var
    path = os.path.join(Data_directory, category) #path to FRUITS dir / the method here concatenate 2 path components with 1 directory separator
    for img in os.listdir(path): #nested loop / 'os.listdir() is a method that reads all files inside FRUITS
        img_array = cv.imread(os.path.join(path, img)) #cv.imread method loads image from the
        # specified path and converts it into a numpy array
        img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB) #converts to a RGB image 
        plt.imshow(img_array) #creates an image from a 2D numpy array
 #       plt.show() #prints image
        break
    break
#attribute from numpy array '.shape' returns tuple representing (height<pixels>, width<pixels>, and number of channels)
print(img_array.shape)

img_size = 40 #variable created to atribute new size to the image
new_array = cv.resize(img_array, (img_size, img_size)) #loaded image, and give the new size of pixels
blur = cv.GaussianBlur(new_array, (5,5),)
plt.imshow(new_array)#creates an image from the 2D array
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.show() #prints image

image_training = [] #creating an empty list

def create_training_image():  #creating function to train data
    for category in Categories: #create variable for Categories-directory<iterable object>
        path = os.path.join(Data_directory, category) #path to bananas or apples dir
        class_num = Categories.index(category) #new variable to store an index of the categories -
        # '.index' method finds the given element in a list and returns its position (starts from 0) 
        for img in os.listdir(path):  #reading all files inside FRUITS
            try:
                img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE) #creating a numpy array from the 2D images
                new_array = cv.resize(img_array, (img_size, img_size)) #(loading images, new pixel size)
                image_training.append([new_array, class_num]) #(array of the resized
                # images, categories index to read the images position) and iterating through all the pictures
            except Exception as e:
                pass

#image_training now stores an array of the images and their position
create_training_image() #calling the function
print(len(image_training)) # print amount of pictures stored in the image_training list


random.shuffle(image_training) #pass image_training through a method to shuffle through the list randomly
#used so the neural network can make better predictions

for sample in image_training[:10]: #variable sample in iterable image_training list
    print(sample[1])#printing sample list 

#variables we are using before we feed it to the neural network 
x = [] #empty list Features
y = [] #empty list labels

for features, label in image_training:
    x.append(features)  #input
    y.append(label)  #comparision 

print(image_training)

#its not possible to pass a list through the neural network, so we need to convert x to a numpy array.

x = np.array(x).reshape(-1, img_size, img_size, 1)
#converting x to an numpy array
#reshaping : -1 = how many features we have (catch all), shape of the data already defined, 1 = grayscale (when changing to RGB this should be 3)

#serializing the object and then writes it to a file. converts the created array into a character stream.
# this character stream will contain all the information to reconstruct the object in another python script
# Using this we can just train the algorithm once, store it to a variable (an object), and then we pickle it.  
pickle_out = open ('x.pickle', 'wb') #(param1 = array, param2 = writing bytes
pickle.dump(x, pickle_out) #put the dict into the opened file
pickle_out.close()#close the file

pickle_out = open ('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()


#accessing the pickled file
pickle_in = open("x.pickle","rb") #here we open the x.pickle file , read bytes
x = pickle.load(pickle_in) #use pickle.load() to load it to a var

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
y = np.array(y)
x = x/255.0  #dividing by the number of pixels to obtain 0 and 1

#here we fed the data through a convolutional neural network to train

model = Sequential()
#(units per layer, kernel size)
model.add(Conv2D(256, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(150, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors

model.add(Dense(1))
model.add(Activation('sigmoid')) #returns a value between 0 and 1

model.compile(loss='binary_crossentropy', #probabilistic loss
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, batch_size=32, epochs=20, validation_split=0.3, callbacks = [tensorboard])


