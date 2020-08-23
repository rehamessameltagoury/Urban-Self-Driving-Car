import argparse
import os
from copy import deepcopy
from pathlib import Path
import shutil
import time
from datetime import datetime as dt
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import keras.backend as K
from keras.models import Model
from keras.layers import BatchNormalization, Dropout, Flatten, Dense, Input, Conv2D, concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.regularizers import l2

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn-dark-palette')
mpl.style.use('seaborn-whitegrid')

from utils import compose_input_for_nn, get_ordered_dict_for_labels

from generator2 import get_data_gen, batcher, remove_speed_labels_from_outputs_spec
from config import (
    IMAGE_DECIMATION,
    IMAGE_SIZE,
    BATCH_SIZE, NUM_EPOCHS,
    NUM_X_CHANNELS, NUM_X_DIFF_CHANNELS,
    TRAIN_SET, TEST_SET,
    WEIGHT_EXPONENT, MIN_SPEED,
    IMAGE_CLIP_UPPER, IMAGE_CLIP_LOWER,
    SPEED_AS_INPUT, OUTPUTS_SPEC,
    ERROR_PLOT_UPPER_BOUNDS,
    BASE_FONTSIZE
)

def steer(
    input_shape):
    
    """
    Returns a 2-tuple of (input, embedding_layer) that can later be used
    to create a model that builds on top of the embedding_layer.

    Image normalization to avoid saturation and make gradients work better.
    Convolution: 7x7, filter: 96,  strides: 1x1, activation: RELU
    MaxPooling : pool_size 2x2
    Convolution: 5x5, filter: 256, strides: 1x1, activation: RELU
    BatchNormalization
    MaxPooling : pool_size 2x2
    Convolution: 3x3, filter: 384, strides: 1x1, activation: RELU
    Convolution: 3x3, filter: 384, strides: 1x1, activation: RELU
    Convolution: 3x3, filter: 256, strides: 1x1, activation: RELU
    Fully connected: neurons: 1500, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 50, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 50, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 1 (output)
    # the convolution layers: are meant to handle feature engineering
    the fully connected layer: for predicting the steering angle.
    the maxpooling layer : to progressively reduce the spatial size of the representation
                           to reduce the amount of parameters and computation in the network.
    BatchNormalization: to Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.                       
    dropout: avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    RELU(rectified linear activation function) function lead to very high-performance networks. 
    This function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive.
    """

    x = inp = Input(input_shape)
    x = Conv2D(96, (7, 7) , activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256, (5, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(384, (3, 3), activation='relu')(x)
    x = Conv2D(384,(3,3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1500, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='elu')(x)
    enc = Dropout(0.5)(x)
    
    return inp, enc  #enc=steer

def add_throttle(
    outputs_spec, embed_getter,
    act='elu', l2_reg=1e-3, num_dense_neurons=512, #512
):
    #Build output layers on top of the embedding layer, and return a model which predict steer and throttle.
    inp, emb = embed_getter()
    #2nd input = speed sequence
    inp_speed = Input((1, ), name='speed')

    # First, each steering angle gets its own prediction neuron
    steer_outputs = []
    for layer_name in outputs_spec.keys():
        if 'steer' in layer_name:
            steer_outputs.append(
                Dense(
                    1,
                    kernel_regularizer=l2(l2_reg),
                    activation=outputs_spec[layer_name]['act'],#linear function
                    name=layer_name,
                )(emb)
            )

    # Now, we concatenate the embedding layer with the speed (provided as input)
    # and the outputs for the steering angles
    emb = concatenate([emb, inp_speed] + steer_outputs)
    throttle_outputs = []
    for layer_name in outputs_spec.keys():
        if 'throttle' in layer_name:
            throttle_outputs.append(
                Dense(
                    1,
                    kernel_regularizer=l2(l2_reg),
                    activation=outputs_spec[layer_name]['act'],
                    name=layer_name,
                )(emb)
            )

    return Model([inp, inp_speed], steer_outputs+throttle_outputs)


def extract_y(filename):
    relevant_component = filename.split('/')[1].split('_depth_data')[0]
    episode = filename.split('_depth_data')[1].split('.npy')[0]
    DF_log = pd.read_csv('logs/{}_log{}.txt'.format(relevant_component, episode))
    if 'speed' in DF_log:
        which_OK = (DF_log['speed'] > MIN_SPEED)
        speed = DF_log[which_OK]['speed']
    else:
        which_OK = DF_log.shape[0] * [True]
        speed = pd.Series(DF_log.shape[0] * [-1])
    steer = DF_log[which_OK]['steer']
    throttle = DF_log[which_OK]['throttle']
    return which_OK, steer, throttle, speed


def _get_data_from_one_racetrack(filename):
    which_OK, steer, throttle, speed = extract_y(filename)
    X = pd.np.load(filename)[..., which_OK].transpose([2, 0, 1])

    if X.shape[1] != (IMAGE_CLIP_LOWER-IMAGE_CLIP_UPPER) // IMAGE_DECIMATION:
        X = X[:, IMAGE_CLIP_UPPER:IMAGE_CLIP_LOWER, :][:, ::IMAGE_DECIMATION, ::IMAGE_DECIMATION]

    # Need to expand dimensions to be able to use convolutions
    X = np.expand_dims(X, 3)

    return X, {
        'steer': steer.values,
        'throttle': throttle.values,
        'speed': speed.values
    }


def get_data(filenames):
    X_all = []
    labels_all = []
    racetrack_labels = []
    for filename in filenames:
        X, labels = _get_data_from_one_racetrack(filename)
        X_all.append(X)
        labels_all.append(labels)
        racetrack_index = int(filename.split('racetrack')[1].split('_')[0])
        racetrack_labels += len(labels)*[racetrack_index]

    label_names = labels.keys()
    X_out = np.concatenate(X_all)
    labels_out = {
        label_name: np.concatenate([labels[label_name] for labels in labels_all])
        for label_name in label_names
    }
    labels_out['racetrack'] = pd.get_dummies(racetrack_labels).values #convert catogerals variables into indecated variables 
    return X_out, labels_out


def generator_fit(
    model, X, labels, weights, num_X_channels, num_Xdiff_channels, batch_size,
    outputs_spec, speed_as_input, verbosity=2  #num of epoch appears like this "Epoch 1/10"
):
    train_gen = get_data_gen(
        X, labels, weights,
        num_X_channels, num_Xdiff_channels, outputs_spec,
        validation=False,
        flip_prob=0.5
    )
    train_batch_gen = batcher(train_gen, batch_size, outputs_spec, speed_as_input)

    valid_gen = get_data_gen(
        X, labels, np.ones(X.shape[0]),
        num_X_channels, num_Xdiff_channels, outputs_spec,
        validation=True,
        flip_prob=0.0
    )
    valid_batch_gen = batcher(valid_gen, batch_size, outputs_spec, speed_as_input)

    epoch_size = 4*X.shape[0]  # To account for mirror reflections

    model.fit_generator(
        train_batch_gen,
        steps_per_epoch=epoch_size//batch_size,
        epochs=1,
        validation_data=valid_batch_gen,
        validation_steps=0.01*epoch_size,
        verbose=verbosity
    )

    return model


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-r', '--results_dir',
        default=dt.today().strftime('%Y-%m-%d-%H-%M'),
        dest='results_dir',
        help='Directory for storing the results')

    args = argparser.parse_args()

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    shutil.copy('config.py', results_dir)
    (Path(results_dir) / '__init__.py').touch()

    input_shape = (
        (IMAGE_CLIP_LOWER-IMAGE_CLIP_UPPER) // IMAGE_DECIMATION,
        IMAGE_SIZE[1] // IMAGE_DECIMATION,
        NUM_X_CHANNELS+NUM_X_DIFF_CHANNELS
    )

    embed_getter = lambda: steer(input_shape)

    
    model = add_throttle(OUTPUTS_SPEC, embed_getter)

    loss = {layer_name: spec['loss'] for layer_name, spec in OUTPUTS_SPEC.items()}
    loss_weights = {layer_name: spec['weight'] for layer_name, spec in OUTPUTS_SPEC.items()}
    if SPEED_AS_INPUT:
        for layer_name in OUTPUTS_SPEC.keys():
            if 'speed' in layer_name:
                del loss[layer_name]
                del loss_weights[layer_name]
    model.compile(
        loss=loss,
        loss_weights=loss_weights,
        optimizer=Adam(1e-5),
    )

    # Get the test set
    X_test, labels_test = get_data(TEST_SET)

    errors_storage = {key: [] for key in OUTPUTS_SPEC}
    min_error = 10000
    for epoch in range(1, NUM_EPOCHS+1):
        weights = None
        for episode, filename in enumerate(TRAIN_SET):
            print('EPOCH: {}, EPISODE: {}'.format(epoch, episode))

            print('Getting data...')
            start = time.time()
            X, labels = get_data([filename])
            print('Took {:.2f}s'.format(time.time() - start))

            if weights is None:
                weights = np.ones(X.shape[0])
            else:
                print('Calculating weights...')
                start = time.time()
                if WEIGHT_EXPONENT == 0:
                    weights = np.ones(X.shape[0])
                else:
                    # can be done more efficiently
                    preds = np.zeros(X.shape[0])
                    num_channels_max = max(NUM_X_CHANNELS, NUM_X_DIFF_CHANNELS+1)
                    for index in range(num_channels_max, X.shape[0]):
                        slc = slice(index-num_channels_max, index+1)
                        preds[slc] = model.predict(
                            compose_input_for_nn(X[slc], NUM_X_CHANNELS, NUM_X_DIFF_CHANNELS)
                        )[0] 
                    weights = np.abs(preds - labels['steer'])**WEIGHT_EXPONENT
                print('Took {:.2f}s'.format(time.time() - start))

            model = generator_fit(
                model, X, labels, weights,
                NUM_X_CHANNELS, NUM_X_DIFF_CHANNELS, BATCH_SIZE,
                OUTPUTS_SPEC, SPEED_AS_INPUT
            )

        if epoch % 5 == 0:
            model.save('{}/model{}.h5'.format(results_dir, epoch))




if __name__ == '__main__':
    main()