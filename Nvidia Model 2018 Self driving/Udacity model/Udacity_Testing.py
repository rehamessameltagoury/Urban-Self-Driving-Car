#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO

#load our saved model
from keras.models import load_model

#helper class
import utils
def Append_Sequence(listt , value , length):
    for i in range(0,length):
        listt[:,length-1-i,:] = listt[:,length-1-i-1,:]
    listt[:,0,:] = value
    return listt

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
model2 = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 30
MIN_SPEED = 0
speed_sequence = np.zeros((1,10,1))
#and a speed limit
speed_limit = MAX_SPEED

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        global speed_sequence
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array
            speed_sequence = np.array(speed_sequence)
            # predict the steering angle for the image
            (Predicted_speed ,steering_angle) = model.predict([image,speed_sequence], batch_size=1)
            #steering_angle = float(model.predict(image, batch_size=1))

            #print("steering_angle = ",str(steering_angle))
            #print("steering_angle_type = ",type(steering_angle) )
            #print("Predicted_speed = ",str(Predicted_speed) )
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            predicted_steer = steering_angle[0][0] -1.0
            acceleration_type = 0
            for i in range(1,3):
                temp = Predicted_speed[0][0]
                if temp < Predicted_speed[0][i]:
                    temp = Predicted_speed[0][i]
                    acceleration_type = i
                

            if acceleration_type == 0:
                # decelrate
                speed -= 1
                print("Decelrating-----------------------------------")
            elif acceleration_type == 2:
                #accelere 
                speed += 1
                print("------------------------------------Accelrating")
            else: 
                print("-------------------Maintain--------------------")
            if predicted_steer>0:
                print("-------------------------------------turn right")
            elif predicted_steer<0:
                print("turn left--------------------------------------")
            else:
                print("--------------------Center---------------------")

            speed_sequence = Append_Sequence(speed_sequence ,float(speed) , 10)

            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - predicted_steer**2 - (speed/speed_limit)**2

            #print('{} {} {}'.format(steering_angle, throttle, speed))
            print("Speed = " , str(format(speed, '.4f')))
            print("Steering angle = ", str(format(predicted_steer, '.4f')))
            print("throttle = " , str(format(throttle, '.4f')))
            send_control(predicted_steer, throttle,speed)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0 , 0)


def send_control(steering_angle, throttle,speed):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            'speed':speed.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model
    model = load_model(args.model)
    #model2 = load_model('model2.h5')


    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
