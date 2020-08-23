import numpy as np
import importlib
import keras
import carla 
from utils import clip_throttle, compose_input_for_nn
from abstract_controller import Controller


class NNController(Controller):
    def __init__(self, target_speed, model_dir_name, which_model,
                 ensemble_prediction,
        ):
        self.target_speed = target_speed
        
        self.predict_throttle = True 
        # read config.py and read variables 
        config = importlib.import_module(model_dir_name + '.config')
        if self.predict_throttle:
            self.throttle_index = [
                i for i, key in enumerate(config.OUTPUTS_SPEC) if key == 'throttle'
            ][0]
        self.speed_as_input = config.SPEED_AS_INPUT
        self.predict_throttle = ('throttle' in config.OUTPUTS_SPEC)
        self.num_X_channels = config.NUM_X_CHANNELS
        self.num_Xdiff_channels = config.NUM_X_DIFF_CHANNELS
        self.num_channels_max = max(self.num_X_channels, self.num_Xdiff_channels+1)

        if which_model == 'best':
            which_model = '_best'

        model_file_name = '{model_dir_name}/model{which_model}.h5'.format(
            model_dir_name=model_dir_name,
            which_model=which_model
        )
        # read model
        self.model = keras.models.load_model(model_file_name)

        self.prev_depth_arrays = []
        self.steer = 0
        self.throttle = 1
        

        self.ensemble_prediction = ensemble_prediction
        # save indexes of steer and throttle
        if self.ensemble_prediction:
            steps_into_future = config.STEPS_INTO_NEAR_FUTURE
            self.actuator_indexes = {
                'steer': [0] + list(steps_into_future),
                'throttle': [self.throttle_index] + [self.throttle_index + i for i in steps_into_future],
            }
            num_preds = len(steps_into_future) + 1
            self.past_preds = {
                'steer': np.zeros((num_preds, num_preds)),
                'throttle': np.zeros((num_preds, num_preds)),
            }

    def control(self, pts_2D, measurements, depth_array):
        which_closest, _, _ = self._calc_closest_dists_and_location(
            measurements,
            pts_2D
        )

        curr_speed = measurements.player_measurements.forward_speed * 3.6

        depth_array = np.expand_dims(np.expand_dims(depth_array, 0), 3)

        self.prev_depth_arrays.append(depth_array)

        if len(self.prev_depth_arrays) < self.num_channels_max:
            pred = None
        else:
            self.prev_depth_arrays = self.prev_depth_arrays[-self.num_channels_max:]
            X_full = np.concatenate(self.prev_depth_arrays[-self.num_channels_max:])
            X = compose_input_for_nn(X_full,self.num_X_channels, self.num_Xdiff_channels)


            # predict steer and throttle from model or steer only
            if self.speed_as_input:
                pred = self.model.predict([X, np.array([curr_speed])])
            else:
                pred = self.model.predict(X)

        # take the value of steer and throttle and update the current value of car
        if pred is not None:
            if self.predict_throttle:
                curr_steer_pred = pred[0][0, 0]
                curr_throttle_pred = pred[self.throttle_index][0, 0]

                if self.ensemble_prediction:
                    curr_steer_pred = self._ensemble_prediction(pred, 'steer')
                    curr_throttle_pred = self._ensemble_prediction(pred, 'throttle')

                self.steer = curr_steer_pred
                self.throttle = curr_throttle_pred
                
            else:
                curr_steer_pred = pred[0][0, 0] 
                
                self.steer = curr_steer_pred

                self.throttle = clip_throttle(
                    self.throttle,
                    curr_speed,
                    self.target_speed
                )
        
        # situations the car faces
        for agent in measurements.non_player_agents:
            
            agent.id # unique id of the agent

            # situation 1 (interact with other cars)
            if agent.HasField('vehicle'):
                agent.vehicle.forward_speed
                agent.vehicle.transform
               
                # To get the closest car
                if (measurements.player_measurements.transform.orientation.x > 0.9 or measurements.player_measurements.transform.orientation.y > 0.9 ):
                    if (( ( (agent.vehicle.transform.location.x) -  (measurements.player_measurements.transform.location.x) > 5 ) and ( (agent.vehicle.transform.location.x) - (measurements.player_measurements.transform.location.x) < 15) and  (abs(measurements.player_measurements.transform.location.y - agent.vehicle.transform.location.y ) < 3) and  (abs(measurements.player_measurements.transform.location.y - agent.vehicle.transform.location.y) > 0)) or  ( ( (agent.vehicle.transform.location.y) - (measurements.player_measurements.transform.location.y) > 5) and ( (agent.vehicle.transform.location.y) - (measurements.player_measurements.transform.location.y) < 15) and  (abs(measurements.player_measurements.transform.location.x - agent.vehicle.transform.location.x ) < 3) and  (abs(measurements.player_measurements.transform.location.x - agent.vehicle.transform.location.x) > 0))):    
                        
                        # closest car stoped
                        if (agent.vehicle.forward_speed < 2 ) :
                            self.throttle = 0.0
                            self.brake = 1.0
                            self.steer = 0.0

                        # closest car has speed so we decrease our car speed
                        elif(agent.vehicle.forward_speed > 0):
                            curr_speed2 = measurements.player_measurements.forward_speed * 3.6
                            target_speed2 = agent.vehicle.forward_speed - 10
                            self.throttle = clip_throttle(
                                self.throttle,
                                curr_speed2,
                                target_speed2
                            )

                elif (measurements.player_measurements.transform.orientation.x < -0.9 or measurements.player_measurements.transform.orientation.y < -0.9):
                    if (( ((measurements.player_measurements.transform.location.x) - (agent.vehicle.transform.location.x) > 5 ) and ((measurements.player_measurements.transform.location.x) - (agent.vehicle.transform.location.x) < 15) and  (abs( agent.vehicle.transform.location.y -measurements.player_measurements.transform.location.y ) < 3) and  (abs( agent.vehicle.transform.location.y-measurements.player_measurements.transform.location.y ) > 0)) or  ( ( (measurements.player_measurements.transform.location.y) - (agent.vehicle.transform.location.y)  > 5) and ( (measurements.player_measurements.transform.location.y) - (agent.vehicle.transform.location.y)  < 15) and  (abs(measurements.player_measurements.transform.location.x - agent.vehicle.transform.location.x ) < 3) and  (abs(measurements.player_measurements.transform.location.x - agent.vehicle.transform.location.x) > 0))):    
                        if (agent.vehicle.forward_speed < 2 ) :
                            self.throttle = 0.0
                            self.brake = 1.0
                            self.steer = 0.0
                        elif(agent.vehicle.forward_speed > 0):
                            curr_speed2 = measurements.player_measurements.forward_speed * 3.6
                            target_speed2 = agent.vehicle.forward_speed - 10
                            self.throttle = clip_throttle(
                                self.throttle,
                                curr_speed2,
                                target_speed2
                            )
                        

            # situation 2 (inteact with traffic lights)  
            if agent.HasField('traffic_light'):
                agent.traffic_light.transform
                agent.traffic_light.state

                # closest traffic light
                if (measurements.player_measurements.transform.orientation.x < -0.9 or measurements.player_measurements.transform.orientation.y < -0.9 ):
                   if (( ((measurements.player_measurements.transform.location.x) - (agent.traffic_light.transform.location.x) > 2 ) and ((measurements.player_measurements.transform.location.x) - (agent.traffic_light.transform.location.x) < 15) and  (abs( agent.traffic_light.transform.location.y -measurements.player_measurements.transform.location.y ) < 3) and  (abs( agent.traffic_light.transform.location.y-measurements.player_measurements.transform.location.y ) > 0)) or  ( ( (measurements.player_measurements.transform.location.y) - (agent.traffic_light.transform.location.y)  > 2) and ( (measurements.player_measurements.transform.location.y) - (agent.traffic_light.transform.location.y)  < 15) and  (abs(measurements.player_measurements.transform.location.x - agent.traffic_light.transform.location.x ) < 3) and  (abs(measurements.player_measurements.transform.location.x - agent.traffic_light.transform.location.x) > 0))):    
                        if (agent.traffic_light.transform.rotation.yaw - measurements.player_measurements.transform.rotation.yaw < 95 and agent.traffic_light.transform.rotation.yaw - measurements.player_measurements.transform.rotation.yaw > 85):
                            if (agent.traffic_light.state == 1 or agent.traffic_light.state == 2 ) : # 1 --> Red and 2 --> yellow ,, action = stop
                                self.throttle = 0.0
                                self.brake = 1.0
                            

                elif (measurements.player_measurements.transform.orientation.x > 0.9 or measurements.player_measurements.transform.orientation.y > 0.9 ):
                    if (( ( (agent.traffic_light.transform.location.x) -  (measurements.player_measurements.transform.location.x) > 2 ) and ( (agent.traffic_light.transform.location.x) - (measurements.player_measurements.transform.location.x) < 15) and  (abs(measurements.player_measurements.transform.location.y - agent.traffic_light.transform.location.y ) < 3) and  (abs(measurements.player_measurements.transform.location.y - agent.traffic_light.transform.location.y) > 0)) or  ( ( (agent.traffic_light.transform.location.y) - (measurements.player_measurements.transform.location.y) > 2) and ( (agent.traffic_light.transform.location.y) - (measurements.player_measurements.transform.location.y) < 15) and  (abs(measurements.player_measurements.transform.location.x - agent.traffic_light.transform.location.x ) < 3) and  (abs(measurements.player_measurements.transform.location.x - agent.traffic_light.transform.location.x) > 0))):    
                        if (agent.traffic_light.transform.rotation.yaw - measurements.player_measurements.transform.rotation.yaw < 95 and agent.traffic_light.transform.rotation.yaw - measurements.player_measurements.transform.rotation.yaw > 85):
                            if (agent.traffic_light.state == 1 or agent.traffic_light.state == 2 ) :
                                self.throttle = 0.0
                                self.brake = 1.0
                            


            # situation 3 (inteact with pedestrians)
            if agent.HasField('pedestrian'):
                agent.pedestrian.transform
                agent.pedestrian.forward_speed
                
                # closest pedestrian action stop till he passes
                if (measurements.player_measurements.transform.orientation.x > 0.9):
                    if((agent.pedestrian.transform.location.x - measurements.player_measurements.transform.location.x) > 5 and (agent.pedestrian.transform.location.x - measurements.player_measurements.transform.location.x) < 15  and (agent.pedestrian.transform.location.y - measurements.player_measurements.transform.location.y) > -2  and  (agent.pedestrian.transform.location.y - measurements.player_measurements.transform.location.y) <2 ) :
                        self.throttle = 0.0
                        self.brake = 1.0
                    
                elif(measurements.player_measurements.transform.orientation.x < -0.9):
                    if((measurements.player_measurements.transform.location.x -agent.pedestrian.transform.location.x) > 5 and (measurements.player_measurements.transform.location.x - agent.pedestrian.transform.location.x) < 15  and (measurements.player_measurements.transform.location.y - agent.pedestrian.transform.location.y) > -2  and  (measurements.player_measurements.transform.location.y - agent.pedestrian.transform.location.y) <2 ) :
                        self.throttle = 0.0
                        self.brake = 1.0
                    

                elif (measurements.player_measurements.transform.orientation.y > 0.9):
                    if((agent.pedestrian.transform.location.y - measurements.player_measurements.transform.location.y) > 5 and (agent.pedestrian.transform.location.y - measurements.player_measurements.transform.location.y) < 15  and (agent.pedestrian.transform.location.x - measurements.player_measurements.transform.location.x) > -2  and  (agent.pedestrian.transform.location.x - measurements.player_measurements.transform.location.x) <2 ) :
                        self.throttle = 0.0
                        self.brake = 1.0
                        

                elif(measurements.player_measurements.transform.orientation.y < -0.9):
                    if((measurements.player_measurements.transform.location.y -agent.pedestrian.transform.location.y) > 5 and (measurements.player_measurements.transform.location.y - agent.pedestrian.transform.location.y) < 15  and (measurements.player_measurements.transform.location.x - agent.pedestrian.transform.location.x) > -2  and  (measurements.player_measurements.transform.location.x - agent.pedestrian.transform.location.x) <2 ) :
                        self.throttle = 0.0
                        self.brake = 1.0
                       
            
                
                          
        one_log_dict = {
            'steer': self.steer,
            'throttle': self.throttle,
            'which_closest': which_closest,
        }

        return one_log_dict

    def _ensemble_prediction(self, pred, which_actuator):
        self.past_preds[which_actuator] = np.roll(self.past_preds[which_actuator], shift=1, axis=0)
        self.past_preds[which_actuator][0] = [
            pred[i][0, 0] for i in self.actuator_indexes[which_actuator]
        ]
        return np.mean(self.past_preds[which_actuator].diagonal())
