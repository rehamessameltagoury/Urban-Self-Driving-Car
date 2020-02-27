import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
import keras
from keras.models import Model
from keras.layers import Input 
from keras.layers import concatenate
from keras.utils import plot_model

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from sklearn.model_selection import train_test_split #to split out training and testing data
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten , BatchNormalization,LSTM
#helper class to define input shape and generate training images given image paths & steering angles
from utils import INPUT_SHAPE, batch_generator
#for command line arguments
import argparse
#for reading files
import os
#import tensorflow as tf
#from tensorflow import ConfigProto

#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)
data_df=[]
S=[]

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
import tensorflow as tf
 

from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
'''                                         ### THE OLD LOAD DATA ONE ###
def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'comdataandpix.csv'), names=['index','center', 'speed' ,'steering' ,'Throttle','brake'])

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df['center'].values
    #and our steering commands as our output data
    y = data_df['steering'].values
    #S = data_df['speed'].values

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    X_valid = X_train[60:]
    X_train = X_train[:60]
    y_valid = y_train[60:]
    y_train = y_train[:60]
    return X_train, X_valid, y_train, y_valid
    '''
                                            ### THE NEW LOAD DATA ONE ###
def load_data(args):
    #Load training data and split it into training and validation set
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'comdataandpix.csv'), names=['index','center', 'speed' ,'steering' ,'Throttle','brake', 'Speed_Sequence'])
    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    x_Images = data_df['center'].values
    x_Speed_Sequence  = data_df['Speed_Sequence'].values
    #and our speed commands as our output data
    y_steer = data_df['steering'].values
    y_speed = data_df['speed'].values
    #S = data_df['speed'].values
    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train_image, X_valid_image, Y_train_steer, Y_valid_steer = train_test_split(x_Images, y_steer, test_size=args.test_size, random_state=0)
    X_train_Sequence, X_valid_Sequence, Y_train_speed, Y_valid_speed = train_test_split(x_Speed_Sequence, y_speed, test_size=args.test_size, random_state=0)
    '''X_valid = X_train[60:]
    X_train = X_train[:60]
    y_valid = y_train[60:]
    y_train = y_train[:60]
    '''
    #X_train, X_valid, y_train, y_valid = train_test_split(x_Images, y, test_size=args.test_size, random_state=0)
    #return X_train, X_valid, y_train, y_valid
    return  X_train_image, X_valid_image, Y_train_steer, Y_valid_steer , X_train_Sequence, X_valid_Sequence, Y_train_speed, Y_valid_speed  #, y_train_steer, y_valid_steer
    #return X_train, X_valid, y_train, y_valid
def build_model_speed(args):
    #steer model 
    Steer_In =Input(shape=(227,227,3),name='model1_in')
    #conv1
    Steer = Conv2D(96, 11, 11, activation='relu', subsample=(4,4))(Steer_In)
    Steer = BatchNormalization()(Steer)
    Steer = MaxPooling2D(pool_size=(3,3),strides=(2,2))(Steer)
    #conv2
    Steer = Conv2D(256, 5, 5, activation='relu', subsample=(1,1))(Steer)
    Steer = BatchNormalization()(Steer)
    Steer = MaxPooling2D(pool_size=(3,3),strides=(2,2))(Steer)
    #conv3
    Steer = Conv2D(384, 3, 3, activation='relu', subsample=(1,1))(Steer)
    #conv4
    Steer = Conv2D(384,3,3, activation='relu', subsample=(1,1))(Steer)
    #conv5
    Steer = Conv2D(256, 3, 3, activation='relu',subsample=(1,1))(Steer)
    Steer = Flatten()(Steer)
    #FC1
    Steer = Dense(1024, activation='elu')(Steer)
    Steer = Dropout(args.keep_prob , name='Dropout1')(Steer)
    #FC2
    Steer = Dense(50, activation='elu')(Steer)
    Steer_out = Dropout(args.keep_prob, name='Dropout2')(Steer)
    Steer_Model1 = Model(inputs=Steer_In, outputs=Steer_out)
    ################################################################################
    ##############################speed model2 before concate_model#################
    Speed_In = Input(shape=(10,1),name='Speed_In')
    Speed =LSTM(128,return_sequences=True)(Speed_In)
    Speed = Flatten()(Speed)
    #model2_in = Embedding(len(word_index) + 1, 300, input_length = 40, dropout = 0.2)
    #L2_out = (LSTM(300, dropout_W = 0.2, dropout_U = 0.2)(L2_out)
    #L2_out = Flatten()(L2_out)
    #FC1
    Speed = Dense(50,activation='elu')(Speed)
    Speed = Dropout(args.keep_prob, name='Dropout3')(Speed)
    #FC2
    Speed = Dense(50, activation='elu')(Speed)
    Speed = Dropout(args.keep_prob, name='Dropout4')(Speed)
    Speed_Model1 = Model(inputs=Speed_In, outputs=Speed)
    ################ here is our concate_model#########################
    #concatenate(inputs, axis=-1, **kwargs):
    First_Merge = concatenate([Speed_Model1.output, Steer_Model1.output], name='Concatenate')
    #print(merged_layers)           
    #out = BatchNormalization()(merged_layers)
    Speed_Continue = Dense(50, activation='elu')(First_Merge)
    Speed_Continue = Dropout(args.keep_prob, name='Dropout5')(Speed_Continue)
    Speed_Out = Dense(1, name='Dense_last_speed')(Speed_Continue)
    Speed_Model = Model(inputs = [Steer_In,Speed_In],outputs = [Speed_Out])
    #continue Steering model
    Steer_Continue = Dense(50, activation='elu')(Steer_out) #node 1 output to FC3
    Steer_Continue = Dropout(args.keep_prob, name='Dropout6')(Steer_Continue) #steer output
    Steer_Continue = Dense(1 , name = 'Dense_last_steer')(Steer_Continue)
    Steer_Model   = Model(inputs = [Steer_In],outputs = [Steer_Continue])

    Final_Model = Model(inputs=[Steer_In , Speed_In], outputs=[Speed_Model.output , Steer_Model.output])
    ########################################################################
    Final_Model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    #Steer_Model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    Final_Model.summary()
    return Final_Model #, Steer_Model
def train_model(concate_model, args,X_train_image, X_valid_image, Y_train_steer, Y_valid_steer , X_train_Sequence, X_valid_Sequence, Y_train_speed, Y_valid_speed):
    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    #concate_model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    
    concate_model.fit_generator(batch_generator(args.data_dir,X_train_image ,X_train_Sequence,Y_train_steer,Y_train_speed, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid_image,X_valid_Sequence ,Y_valid_steer ,Y_valid_speed, args.batch_size, False),
                        nb_val_samples=len(X_valid_image),
                        callbacks=[checkpoint])
    '''
    inputs_training  = batch_generator(args.data_dir,X_train_image ,X_train_Sequence,Y_train_steer,Y_train_speed, args.batch_size, True)
    inputs_Validation  = batch_generator(args.data_dir,X_valid_image ,X_valid_Sequence,Y_valid_steer,Y_valid_speed, args.batch_size, False)

    
    concate_model.fit_generator([inputs_training[0] , inputs_training[1]], [inputs_training[3] , inputs_training[4]],
                    args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=([inputs_Validation[0] , inputs_Validation[1]], [inputs_Validation[0] , inputs_Validation[1]]),
    callbacks=[checkpoint])
    
    
    model.fit(a_train, [x_train, y_train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(a_test, [x_test, y_test]),
              callbacks=[checkpoint])
    score = model.evaluate(a_test, [x_test, y_test], verbose=0) 
    
    concate_model.fit([X_train_image, X_train_Sequence], [Y_train_steer, Y_train_speed],
                    validation_data=([X_valid_image, X_valid_Sequence], [Y_train_steer, Y_train_speed])
          epochs=1, batch_size=10)
    
    '''
#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='D:/Graduation project/CARLA/PythonAPI/examples/_out')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=50)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=1)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)
    #build model
    Total_Model  = build_model_speed(args)        #Steer_Model
    print("Done")
    #train model on data, it saves as model.h5 

    train_model(Total_Model, args, *data)
    #Speed_Model.save("model.h5")


if __name__ == '__main__':
    main()

