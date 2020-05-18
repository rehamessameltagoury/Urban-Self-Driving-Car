
from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import time
import glob
import os
import sys
import pandas as pd 
import csv
import cv2
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from sklearn.model_selection import train_test_split 
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import tensorflow as tf

############################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
############################
from sklearn.metrics import accuracy_score
import base64
import shutil
#matrix math
import numpy as np
#real-time server
#import socketio
#concurrent networking 
#import eventlet
#web server gateway interface
#import eventlet.wsgi
#image manipulation
import PIL
from PIL import Image
#web framework
#from flask import Flask
#input output
from io import BytesIO

#load our saved model
#from keras.models import load_model
#from skimage.util import img_as_float
from tensorflow.keras.models import load_model
from keras.preprocessing import image as img2

#from resizeimage import resizeimage
from numpy.random import seed
from numpy.random import randn
import random
def Append_Sequence(listt , value):
    for i in range(0,9):
        listt[i] = listt[i+1]
    listt[9] = value
    return listt
def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[115:, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    #image = crop(image)
    #image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    '''
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    '''
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def str_to_float(inp):
    inp = inp[1:-1]
    res = [float(idx) for idx in inp.split(',')]
    return res
def random_shadow(image):
    """
    Generates and adds random shadow
    """
    #image = crop(image)
    #image = resize(image)
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    #image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def load_data():
    data_df = pd.read_csv(os.path.join(os.getcwd(), 'D:/Graduation project/CARLA/PythonAPI/examples/speedClassificationModel', 'comdataandpix.csv'), names=['index','left' ,'right','center', 'speed' ,'steering' ,'Throttle','brake' , 'Speed_Sequence','Speed_Classes' ])
    index  = data_df['index'].values
    x_Images = data_df[['center', 'left', 'right']].values
    x_Speed_Sequence  = data_df['Speed_Sequence'].values
    #and our speed commands as our output data
    #sio = StringIO(data_df)
    #pd.read_csv(sio, dtype={"user_id": int, "username": object}) 
    y_steer = data_df['steering'].values
    y_speed = data_df['Speed_Classes'].values
    X_train_image, X_valid_image, Y_train_steer, Y_valid_steer =train_test_split(x_Images, y_steer, test_size=0.1, random_state=46,shuffle= True)
    X_train_Sequence, X_valid_Sequence, Y_train_speed, Y_valid_speed = train_test_split(x_Speed_Sequence, y_speed, test_size=0.1, random_state=46,shuffle= True)

    return  X_valid_image , X_valid_Sequence, Y_valid_steer, Y_valid_speed



acceleration_types = ['decelration' , 'maintain', 'acceleration' ]
dirr = 'D:/Graduation project/CARLA/PythonAPI/examples/speedClassificationModel'
speed_sequence = np.zeros((1,10,1))
#model=load_model(str(input("Please enter name of .h5 file : ")))
model = load_model('modelBest-020.h5')
image=np.array((227,227,3))

#------------------------------


'''

speed_sequence = np.zeros((1,10,1))
image = np.zeros((1,227,227,3))
Predicted_speed,Predicted_steering     = model.predict([image , speed_sequence] ,batch_size = 1)
#steering_angle = float("{:.3f}".format(Predicted_steering[0][0]))


print("Predicted steer= "+str(float(Predicted_steering[0][0])))
print("Predicted speed 'Decelerate prob.' = " +str(float(Predicted_speed[0][0])))
print("Predicted speed 'Maintain prob.' = "+str(float(Predicted_speed[0][1])))
print("Predicted speed 'accelerate prob.' = " +str(float(Predicted_speed[0][2])))

image = np.ones((1,227,227,3))
speed_sequence = np.ones((1,10,1))
#image = np.array([image])
Predicted_speed,Predicted_steering     = model.predict([image , speed_sequence] ,batch_size = 1)
#steering_angle = float("{:.3f}".format(Predicted_steering[0][0]))


print("Predicted steer= "+str(float(Predicted_steering[0][0])))
print("Predicted speed 'Decelerate prob.' = " +str(float(Predicted_speed[0][0])))
print("Predicted speed 'Maintain prob.' = "+str(float(Predicted_speed[0][1])))
print("Predicted speed 'accelerate prob.' = " +str(float(Predicted_speed[0][2])))
'''
def Testing(data_dir, image_paths, speed_sequences , steering_angles,speeds ,Dataset_size):
    """
    Generate training image give image paths and associated steering angles
    """
    global model
    global acceleration_types
    hit_speed = 0
    steering_angle_results = []
    xx = 0
    yy = 0
    ii = 0
    speeds_result = []
    images = np.empty([ 227, 227, 3])
    speed_seq = np.empty((1,10,1))
    speed     = np.empty((1,3))
    throttle = 0
    for k in range(Dataset_size):
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            #print(center)
            steering_angle = steering_angles[index]
            speed_index        = str_to_float(speeds[index])
            speed_seqq       = str_to_float(speed_sequences[index])
            image, steering_angle = augument(data_dir, center, left, right, steering_angle) 
            image = preprocess(image)
            steer = float("{:.3f}".format((steering_angle)))
            speed_sequence = np.array(speed_seqq)
            speed_sequence = speed_sequence.reshape((1,10,1))
            Speed_Classes = np.array(speed_index)
            image = image.reshape((1,227,227,3))

            now = time.time()
            Predicted_speed,Predicted_steering     = model.predict([image , speed_sequence] ,batch_size = 1)
            print("delay of Total_Model predication = "+str("{0:.2f}".format(round((time.time()-now),2)))+ "Seconds") 
            Predicted_steering = float("{:.3f}".format((Predicted_steering[0][0]-1)))

            print("predicted speed classes " +str(Predicted_speed[0]))
            print("predicted steer " +str(Predicted_steering))

            print("Actual Steer = "+str(steer))


            if Predicted_steering > steer:
            	steer_Testing_error = round(Predicted_steering -  steer , 3)
            else:
            	steer_Testing_error = round(steer - Predicted_steering ,3)


            predicted_throttle = Predicted_speed[0][0]	
            actual_throttle = Speed_Classes[0]

            
            for j in range(1,3):
                if predicted_throttle < Predicted_speed[0][j]:
                    predicted_throttle = j
                if actual_throttle < Speed_Classes[j]:
                    actual_throttle = j 
            predicted_throttle = round(predicted_throttle)
            actual_throttle =round(actual_throttle)
            print("predicted_throttle = "+str(predicted_throttle))
            print("actual_throttle = "+str(actual_throttle))
			
            if (abs(actual_throttle - predicted_throttle == 2.0)):
                speed_Testing_error = 1
                print("                                Fatal speed error")

            elif (abs(actual_throttle - predicted_throttle) == 1.0):# or (actual_throttle - predicted_throttle == -1.0):
                speed_Testing_error = 0.1
            else:
            	speed_Testing_error = 0

            print("predicted speed type = {}".format(acceleration_types[int(predicted_throttle)]))
            print("actual speed type = {}".format(acceleration_types[int(actual_throttle)]))
            print("                                    Steering angle testing error = {} %".format(steer_Testing_error*100))
            print("                                    speed padel    testing error = {} %".format(speed_Testing_error*100))
            print('-------------------------------------------------------------------------------------------------')
            steering_angle_results.append(float(1.0- steer_Testing_error))

            speeds_result.append(float(1.0- speed_Testing_error))
            xx += float(1.0- steer_Testing_error)
            yy += float(1.0- speed_Testing_error)
            ii+=1
            if i == Dataset_size:
                break
    steer_testerror = sum(steering_angle_results)/Dataset_size
    speed_testerror = sum(speeds_result)/Dataset_size
    print("steering angle testing accuracy = {}".format(xx/ii))
    print("speed testing accuracy = {}".format(yy/ii))

'''old method'''
x,y,z,w = load_data()
Testing(dirr , x,y,z,w , 20)


'''new  method
x_Images , x_Speed_Sequence, y_steer, y_speed = load_data()

Predicted_speed , Predicted_steering = model.predict([x_Images , x_Speed_Sequence])

acc = accuracy_score([Predicted_speed , Predicted_steering], [y_speed, y_steer])
'''