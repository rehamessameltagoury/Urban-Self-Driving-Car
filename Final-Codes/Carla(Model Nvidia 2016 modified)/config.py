import random
from collections import OrderedDict


# GENERAL
IMAGE_DECIMATION = 4
IMAGE_SIZE = (240, 320)
DTYPE = 'float32'

# Controller regularization (Initial value) use in augmentation 
STEER_NOISE = lambda: random.uniform(-0.1, 0.1)
THROTTLE_NOISE = lambda: random.uniform(-0.05, 0.05)

# collect Dataset by using joystick  
CONTROLLER = 'pad'

TRAIN_SET = [
    'depth_data/{}_racetrack{}_depth_data{}.npy'.format(CONTROLLER, racetrack, episode)
    for episode in range(9)
    for racetrack in ['01']
]
TEST_SET = [
    'depth_data/{}_racetrack{}_depth_data{}.npy'.format(CONTROLLER, racetrack, episode)
    for episode in [9]
    for racetrack in ['01']
]


if (set(TRAIN_SET) & set(TEST_SET)) != set():
    print('TRAIN_SET and TEST_SET are not disjoint!!!!!')

# We're throwing away data that occured at low speed
MIN_SPEED = 0

# NN training
BATCH_SIZE = 32
NUM_EPOCHS = 60 #40
WEIGHT_EXPONENT = 0

# INPUT
NUM_X_DIFF_CHANNELS = 0
NUM_X_CHANNELS = 1
IMAGE_CLIP_UPPER = 0
IMAGE_CLIP_LOWER = IMAGE_SIZE[0]

SPEED_AS_INPUT = True

STEPS_INTO_NEAR_FUTURE = range(1, 11)
OUTPUTS_SPEC = OrderedDict(
    [('steer', {'act': 'linear', 'loss': 'mse', 'weight': 1.0})]
    + [('steer__{}__last'.format(i), {'act': 'linear', 'loss': 'mse', 'weight': 1.0}) for i in STEPS_INTO_NEAR_FUTURE]

    + [('throttle', {'act': 'sigmoid', 'loss': 'mse', 'weight': 1.0})]
    + [('throttle__{}__last'.format(i), {'act': 'sigmoid', 'loss': 'mse', 'weight': 1.0}) for i in STEPS_INTO_NEAR_FUTURE]

)
if SPEED_AS_INPUT:
    OUTPUTS_SPEC['speed'] = {'act': 'linear', 'loss': 'mse', 'weight': 1e-2}

# PLOTTING
BASE_FONTSIZE = 14

ERROR_PLOT_UPPER_BOUNDS = {
    key: {
        'steer': 0.15,
        'throttle': 0.5,
        'speed': 40,
        'racetrack': 1.0,
    }[key.split('__')[0]]
    for key in OUTPUTS_SPEC
}
