import numpy as np
from collections import OrderedDict

import sys
sys.path.append('..')

from carla.util import print_over_same_line

# reach the target speed by clipping throttle 
def clip_throttle(throttle, curr_speed, target_speed):
    return np.clip(
        throttle - 0.01 * (curr_speed-target_speed),
        0.4,
        0.9
    )


def get_ordered_dict_for_labels(outputs_spec, shape):
    return OrderedDict([
        (key, np.zeros(shape)) for key in outputs_spec.keys()
    ])


#This function creates input for a neural network.
def compose_input_for_nn(X_full, num_X_channels, num_Xdiff_channels):
    '''This function creates input for a neural network.
    The input is composed of `num_X_channels` and `num_Xdiff_channels`.

    This function is here (in the utils.py) because it's used by the scripts:
        * neural_network_controller.py
        * train_on_depth.py
        * generator.py
    none of which can claim this functionality is his.
    '''
    X = None if num_X_channels == 0 else X_full[-num_X_channels:]
    if num_Xdiff_channels > 0:
        X_diff = np.diff(
            X_full[-(num_Xdiff_channels+1):],
            axis=0
        )
        X = X_diff if X is None else np.concatenate([X, X_diff])

    # Move channels to the front
    X = np.transpose(X, (3, 1, 2, 0))

    return X
