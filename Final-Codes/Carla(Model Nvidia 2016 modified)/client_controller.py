import os
import argparse
import logging
import random
import time
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev

import sys
sys.path = ['..'] + sys.path

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

from config import (
    IMAGE_SIZE,
    IMAGE_DECIMATION,
    MIN_SPEED,
    DTYPE,
    STEER_NOISE,
    THROTTLE_NOISE,
    IMAGE_CLIP_LOWER,
    IMAGE_CLIP_UPPER
)

from utils import clip_throttle
from gamepad_controller import PadController
import tensorflow as tf

# run it on windows gpu for training 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)




def run_carla_client(args):
    frames_per_episode = 10000
    spline_points = 10000


    track_DF = pd.read_csv('racetrack{}.txt'.format(args.racetrack), header=None)
    # The track data are rescaled by 100x with relation to Carla measurements
    track_DF = track_DF / 100

    pts_2D = track_DF.loc[:, [0, 1]].values  # Speed & Steer values
    #splprep(array , u=these values are calculated automatically as M = len(x[0]), s=A smoothing condition per =1 : data points are considered periodic)
    #tck :tuple () u:array(values of parameters)
    tck, u = splprep(pts_2D.T, u=None, s=2.0, per=1)
    u_new = np.linspace(u.min(), u.max(), spline_points)
    x_new, y_new = splev(u_new, tck, der=0)
    pts_2D = np.c_[x_new, y_new]


    steer = 0.0
    throttle = 0.5

    depth_array = None

    # 'Pad' collect data mode 
    if args.controller_name == 'pad':
        weather_id = 4
        controller = PadController(args.target_speed)
    # 'nn' training mode    
    elif args.controller_name == 'nn':
        # Import it here because importing TensorFlow is time consuming
        from neural_network_controller import NNController  
        weather_id = 4
        controller = NNController(
            args.target_speed,
            args.model_dir_name,
            args.which_model,
            args.ensemble_prediction,
        )
        # connect to server
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')
        episode = 0
        num_fails = 0

        while episode < args.num_episodes:
            # Start a new episode
            depth_storage = None
            log_dicts = None
			
            if args.settings_filepath is None:

                # Here we set the configuration we want for the new episode.
                settings = CarlaSettings()

                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=40,
                    NumberOfPedestrians=150,
                    WeatherId=weather_id,
                    QualityLevel= 'Epic' # lower quality
                )

                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # FOV : is the extent of the observable world that is seen at any given moment
                # 69.4 defualt in intel data sheet 
                camera = Camera('CameraDepth', PostProcessing='Depth', FOV=69.4)
                camera.set_image_size(IMAGE_SIZE[1], IMAGE_SIZE[0])
                camera.set_position(2.30, 0, 1.30)
                settings.add_sensor(camera)

            else:
                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. 
            scene = client.load_settings(settings)

            # Choose one player start at random.
            num_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, num_of_player_starts - 1))

            
            print('Starting new episode...', )
            client.start_episode(player_start)

            status, depth_storage, one_log_dict, distance_travelled = run_episode(
                client,
                controller,
                pts_2D,
                depth_storage,
                log_dicts,
                frames_per_episode,
                args.controller_name
            )

            if 'FAIL' in status:
                num_fails += 1
                print(status)
                continue
            else:
                print('SUCCESS: ' + str(episode))
                episode += 1


def run_episode(client, controller, pts_2D, depth_storage, log_dicts, frames_per_episode, controller_name #, store_data
	):
    num_laps = 0
    curr_closest_waypoint = None
    prev_closest_waypoint = None
    num_waypoints = pts_2D.shape[0]
    num_steps_below_min_speed = 0

    # Iterate every frame in the episode.
    for frame in range(frames_per_episode):
        measurements, sensor_data = client.read_data()
        
        # Read the data produced by the server this frame.
        if measurements.player_measurements.forward_speed * 3.6 < MIN_SPEED:
            num_steps_below_min_speed += 1
        else:
            num_steps_below_min_speed = 0

        too_many_collisions = (measurements.player_measurements.collision_other > 10000)
        too_long_in_one_place = (num_steps_below_min_speed > 300)
        if too_many_collisions:
            return 'FAIL: too many collisions', None, None, None, None
        if too_long_in_one_place:
            return 'FAIL: too long in one place', None, None, None, None


        # read data from camera
        depth_array = np.log(sensor_data['CameraDepth'].data).astype(DTYPE)
        
        # clip the image
        depth_array = depth_array[IMAGE_CLIP_UPPER:IMAGE_CLIP_LOWER, :][::IMAGE_DECIMATION, ::IMAGE_DECIMATION] 
        
        one_log_dict = controller.control(
            pts_2D,
            measurements,
            depth_array,
        )

        prev_closest_waypoint = curr_closest_waypoint
        curr_closest_waypoint = one_log_dict['which_closest']

        # Check if we made a whole lap
        if prev_closest_waypoint is not None:
            # if `0.9*prev_closest_waypoint` is larger than `curr_closest_waypoint`
            # it definitely means that we completed a lap (or the car had been teleported)
            if 0.9*prev_closest_waypoint > curr_closest_waypoint:
                num_laps += 1

        steer, throttle = one_log_dict['steer'], one_log_dict['throttle']

        if controller_name != 'nn':
            # Add noise to "augment" the race
            steer += STEER_NOISE()
            throttle += THROTTLE_NOISE()
        

        client.send_control(
            steer=steer,
            throttle=throttle,
            brake=0.0,
            hand_brake=False,
            reverse=False
        )

    distance_travelled = num_laps + curr_closest_waypoint / float(num_waypoints)
    return 'SUCCESS', depth_storage, one_log_dict, distance_travelled


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    argparser.add_argument(
        '-r', '--racetrack',
        default='01',
        dest='racetrack',
        help='Racetrack number (but with a zero: 01, 02, etc.)')
    argparser.add_argument(
        '-e', '--num_episodes',
        default=10,
        type=int,
        dest='num_episodes',
        help='Number of episodes')
    argparser.add_argument(
        '-s', '--speed',
        default=30,
        type=float,
        dest='target_speed',
        help='Target speed')
    argparser.add_argument(
        '-cont', '--controller_name',
        default='nn',
        dest='controller_name',
        help='Controller name')
    # For the NN controller
    argparser.add_argument(
        '-mf', '--model_dir_name',
        default='model_light',
        dest='model_dir_name',
        help='NN model directory name')
    argparser.add_argument(
        '-w', '--which_model',
        default='35',
        dest='which_model',
        help='Which model to load (5, 10, 15, ..., or: "35")')
   
    argparser.add_argument(
        '-ens', '--ensemble-prediction',
        action='store_true',
        dest='ensemble_prediction',
        help='Whether predictions for steering should be aggregated')

    args = argparser.parse_args()

  

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)




if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
