import pygame
from pygame.locals import *
import sys
from utils import clip_throttle
from abstract_controller import Controller
#import math

throttleCmd = 0.5



class PadController(Controller):
    def __init__(self,target_speed):
        self.target_speed = target_speed
        self.throttle = 0.5
        self.steer = 0.0
        pygame.init()
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
  
   
    def control(self, pts_2D, measurements, depth_array):
        which_closest, _, _ = self._calc_closest_dists_and_location(
            measurements,
            pts_2D #values from csv file (racetrack) after rescalling to be related to carla measurments 
        )

        global throttleCmd

        #event 1) buttonDown 2)buttonUp
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()

            numAxes = self.js.get_numaxes()
            jsInputs = [float(self.js.get_axis(i)) for i in range(numAxes)]
           
            #jsInputs[0]= analog which control the steer, here we check and remove any noise 
            if abs(jsInputs[0]) <= 0.1:
                steerCmd = 0.0
            else:
                steerCmd = jsInputs[0]

            self.steer = steerCmd
          
            curr_speed = measurements.player_measurements.forward_speed * 3.6
            # JOYBUTTONDOWN = pressed (get throttle)
            if event.type == pygame.JOYBUTTONDOWN:
            	
            	#check current speed to be in range (20:25)
            	if curr_speed < 20 :
            		throttleCmd = throttleCmd + 0.1
            	elif curr_speed > 25 :
            		throttleCmd = throttleCmd - 0.1

            	#send throttleCmd to throttle (throttle range (0.0 : 0.8))
            	if throttleCmd <= 0:
            		throttleCmd = 0.0
            	elif throttleCmd > 0.8:
            		throttleCmd = 0.8
            	self.throttle = throttleCmd 
            	
            #JOYBUTTONUP = not pressed >> stopping 
            if event.type == pygame.JOYBUTTONUP:
            	
            	self.throttle = 0.0
            	self.brake = 1.0
            	
            
        
        curr_speed = measurements.player_measurements.forward_speed * 3.6
     
        print('steer: {:.2f}, throttle: {:.2f}'.format(self.steer, self.throttle))

        one_log_dict = {
            'steer': self.steer,
            'throttle': self.throttle,
            'speed': curr_speed,
            'which_closest': which_closest
        }
         
        return one_log_dict   #Data stored in csv file (racetrack)
