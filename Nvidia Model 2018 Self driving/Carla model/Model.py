import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
import keras
from keras.models import Model
from keras.layers import Input 
from keras.layers import concatenate
from keras.utils import plot_model
import matplotlib.pyplot as plt
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split #to split out training and testing data
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
from keras.regularizers import l2
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
from keras import initializers
from tensorflow.keras.models import load_model
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CODA_VISIBLE_DEVICES"] = "0"

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

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
    #data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'comdataandpix.csv'), names=['index','center', 'speed' ,'steering' ,'Throttle','brake', 'Speed_Sequence'])
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'comdataandpix.csv'), names=['index','left' ,'right','center', 'speed' ,'steering' ,'Throttle','brake' , 'Speed_Sequence','Speed_Classes' ])
    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    #x_Images = data_df['center'].values
    index  = data_df['index'].values
    x_Images = data_df[['center', 'left', 'right']].values
    x_Speed_Sequence  = data_df['Speed_Sequence'].values
    #and our speed commands as our output data
    #sio = StringIO(data_df)
    #pd.read_csv(sio, dtype={"user_id": int, "username": object})
    y_steer = data_df['steering'].values
    y_speed = data_df['Speed_Classes'].values



    



    #S = data_df['speed'].values
    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train_image, X_valid_image, Y_train_steer, Y_valid_steer =train_test_split(x_Images, y_steer, test_size=args.test_size, random_state=46,shuffle= True)
    X_train_Sequence, X_valid_Sequence, Y_train_speed, Y_valid_speed = train_test_split(x_Speed_Sequence, y_speed, test_size=args.test_size, random_state=46,shuffle= True)
    '''X_valid = X_train[60:]
    X_train = X_train[:60]
    y_valid = y_train[60:]
    y_train = y_train[:60]
    '''
    '''
    print(X_train_image[0])
    print(X_train_image[1000])

    print(Y_train_speed[0])
    print(Y_train_speed[1000])

    print(X_train_Sequence[0])
    print(X_train_Sequence[1000])


    print(X_train_image[0])
    print(X_train_Sequence[5])
    print(Y_valid_speed[0])

    print(len(X_train_image))
    print(len(Y_valid_steer))
    '''
    #X_train, X_valid, y_train, y_valid = train_test_split(x_Images, y, test_size=args.test_size, random_state=0)
    #return X_train, X_valid, y_train, y_valid
    return  X_train_image, X_valid_image, Y_train_steer, Y_valid_steer , X_train_Sequence, X_valid_Sequence, Y_train_speed, Y_valid_speed  #, y_train_steer, y_valid_steer
    #return X_train, X_valid, y_train, y_valid

'''def build_model_speed(args , Compute_Time = False):
    #steer model 
    Steer_In =Input(shape=(227,227,3),name='model1_in')
    #conv1
    Steer = Conv2D(filters = 96, activation='relu',kernel_size = (11, 11), strides=(4,4),kernel_initializer='he_uniform', bias_initializer='zeros')(Steer_In) #assumed that stride step is 1x1
    Steer = BatchNormalization()(Steer)
    Steer = MaxPooling2D(pool_size=(3,3),strides=(2,2))(Steer)#assumed that stride step is 3x3
    #conv2
    Steer = Conv2D(filters = 256, kernel_size = (5,5), activation='relu', strides=(1,1),kernel_initializer='he_uniform', bias_initializer='zeros')(Steer)
    Steer = BatchNormalization()(Steer)
    Steer = MaxPooling2D(pool_size=(3,3),strides=(2,2))(Steer)#assumed that stride step is 3x3
    #conv3
    Steer = Conv2D(filters =384,kernel_size =(3, 3), activation='relu', strides=(1,1),kernel_initializer='he_uniform', bias_initializer='zeros')(Steer)
    #conv4
    Steer = Conv2D(filters =384,kernel_size =(3,3), activation='relu', strides=(1,1),kernel_initializer='he_uniform', bias_initializer='zeros')(Steer)
    #conv5
    Steer = Conv2D(filters = 256,kernel_size =(3, 3), activation='relu',strides=(1,1),kernel_initializer='he_uniform', bias_initializer='zeros')(Steer)
    Steer = Flatten()(Steer)
    #FC1
    Steer = Dense(1024, activation='relu',kernel_initializer='he_uniform', bias_initializer='zeros')(Steer)
    Steer = Dropout(args.keep_prob , name='Dropout1')(Steer)
    #FC2
    Steer = Dense(50, activation='relu',kernel_initializer='he_uniform', bias_initializer='zeros')(Steer)
    Steer_out = Dropout(args.keep_prob, name='Dropout2')(Steer)
    Steer_Model1 = Model(inputs=Steer_In, outputs=Steer_out)
    ################################################################################
    ##############################speed model2 before concate_model#################
    Speed_In = Input(shape=(10,1),name='Speed_In')
    Speed =LSTM(128,return_sequences=True)(Speed_In)
    Speed = Flatten()(Speed)
    #L2_out = (LSTM(300, dropout_W = 0.2, dropout_U = 0.2)(L2_out)
    #L2_out = Flatten()(L2_out)
    #FC1
    Speed = Dense(50,activation='elu',kernel_initializer='he_uniform', bias_initializer='zeros')(Speed)
    Speed = Dropout(args.keep_prob, name='Dropout3')(Speed)
    #FC2
    Speed = Dense(50, activation='elu',kernel_initializer='he_uniform', bias_initializer='zeros' )(Speed)
    Speed = Dropout(args.keep_prob, name='Dropout4')(Speed)
    Speed_Model1   = Model(inputs=Speed_In, outputs=Speed)
    ################ here is our concate_model#########################
    #concatenate(inputs, axis=-1, **kwargs):
    First_Merge    = concatenate([Speed_Model1.output, Steer_Model1.output], name='Concatenate')
    #print(merged_layers)           
    #out = BatchNormalization()(merged_layers)
    Speed_Continue = Dense(50, activation='elu',kernel_initializer='he_uniform', bias_initializer='zeros')(First_Merge)
    Speed_Continue = Dropout(args.keep_prob, name='Dropout5')(Speed_Continue)
    Speed_Out      = Dense(1, name='Speed',kernel_initializer='he_uniform', bias_initializer='zeros')(Speed_Continue)
    Speed_Model    = Model(inputs = [Steer_In,Speed_In],outputs = [Speed_Out])
    #continue Steering model
    Steer_Continue = Dense(50, activation='relu',kernel_initializer='he_uniform', bias_initializer='zeros')(Steer_out) #node 1 output to FC3
    Steer_Continue = Dropout(args.keep_prob, name='Dropout6')(Steer_Continue) #steer output
    Steer_Continue = Dense(1 , name = 'Steer',kernel_initializer='he_uniform', bias_initializer='zeros')(Steer_Continue)
    Steer_Model    = Model(inputs = [Steer_In],outputs = [Steer_Continue])
    Final_Model    = Model(inputs=[Steer_In , Speed_In], outputs=[Speed_Model.output , Steer_Model.output])
    ########################################################################
    #Steer_Model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    if Compute_Time == True :
        return Steer_Model1  , Speed_Model1 , Speed_Model ,Steer_Model ,Final_Model #, Steer_Model
    else:
        Final_Model.summary()
        return Final_Model
'''
def build_model_speed(args , Compute_Time = False):

    #steer model 
    Steer_In =Input(shape=(227,227,3),name='model1_in')
    #conv1
    #Steer_In = Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE) (Steer)
    #Steer = Lambda(lambda  x: x/127-1.0 )(Steer_In)
    #Steer_In = Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE,name='model1_in')
    Steer = Conv2D(filters = 96, activation='relu',kernel_size = (11, 11), strides=(4,4))(Steer_In) #assumed that stride step is 1x1
    #
    Steer = BatchNormalization()(Steer)
    Steer = MaxPooling2D(pool_size=(3,3),strides=(2,2))(Steer)#assumed that stride step is 3x3
    #conv2
    Steer = Conv2D(filters = 256, kernel_size = (5,5), activation='relu', strides=(1,1))(Steer)
    Steer = BatchNormalization()(Steer)
    Steer = MaxPooling2D(pool_size=(3,3),strides=(2,2))(Steer)#assumed that stride step is 3x3
    #conv3
    Steer = Conv2D(filters =384,kernel_size =(3, 3), activation='relu', strides=(1,1))(Steer)
    #conv4
    Steer = Conv2D(filters =384,kernel_size =(3,3), activation='relu', strides=(1,1))(Steer)
    #conv5
    Steer = Conv2D(filters = 256,kernel_size =(3, 3), activation='relu',strides=(1,1))(Steer)
    Steer = Flatten()(Steer)
    #FC1
    Steer = Dense(1024, activation='relu')(Steer)
    Steer = Dropout(args.keep_prob , name='Dropout1')(Steer)
    #FC2
    Steer = Dense(50, activation='relu')(Steer)
    Steer_out = Dropout(args.keep_prob, name='Dropout2')(Steer)
    Steer_Model1 = Model(inputs=Steer_In, outputs=Steer_out)
    ################################################################################
    ##############################speed model2 before concate_model#################
    Speed_In = Input(shape=(10,1),name='Speed_In')
    Speed =LSTM(128,return_sequences=True)(Speed_In)
    Speed = Flatten()(Speed)
    #L2_out = (LSTM(300, dropout_W = 0.2, dropout_U = 0.2)(L2_out)
    #L2_out = Flatten()(L2_out)
    #FC1
    Speed = Dense(50,activation='elu')(Speed)
    Speed = Dropout(args.keep_prob, name='Dropout3')(Speed)
    #FC2
    Speed = Dense(50, activation='elu' )(Speed)
    Speed = Dropout(args.keep_prob, name='Dropout4')(Speed)
    Speed_Model1   = Model(inputs=Speed_In, outputs=Speed)
    ################ here is our concate_model#########################
    #concatenate(inputs, axis=-1, **kwargs):
    First_Merge    = concatenate([Speed_Model1.output, Steer_Model1.output], name='Concatenate')
    #print(merged_layers)           
    #out = BatchNormalization()(merged_layers)
    Speed_Continue = Dense(50, activation='elu')(First_Merge)
    Speed_Continue = Dropout(args.keep_prob, name='Dropout5')(Speed_Continue)
    Speed_Out      = Dense(3, name='Speed' ,activation='softmax' )(Speed_Continue)
    Speed_Model    = Model(inputs = [Steer_In,Speed_In],outputs = [Speed_Out])
    #continue Steering model
    Steer_Continue = Dense(50, activation='relu')(Steer_out) #node 1 output to FC3
    Steer_Continue = Dropout(args.keep_prob, name='Dropout6')(Steer_Continue) #steer output
    Steer_Continue = Dense(1 , name = 'Steer')(Steer_Continue)
    #Steer_Continue = Lambda(lambda  x: x/127-1.0)(Steer_Continue)
    #Steer_Continue = Lambda(myFunc, output_shape=1)(Steer_Continue)
    Steer_Model    = Model(inputs = [Steer_In],outputs = [Steer_Continue])
    Final_Model    = Model(inputs=[Steer_In , Speed_In], outputs=[Speed_Model.output , Steer_Model.output])
    ########################################################################
    #Steer_Model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    if Compute_Time == True :
        return Steer_Model1  , Speed_Model1 , Speed_Model ,Steer_Model ,Final_Model #, Steer_Model
    else:
        Final_Model.summary()
        return Final_Model

def train_model(concate_model, args,X_train_image, X_valid_image, Y_train_steer, Y_valid_steer , X_train_Sequence, X_valid_Sequence, Y_train_speed, Y_valid_speed ,Continue = 0):
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
                                 save_best_only=False,
                                 mode='auto')
    checkpoint2 = ModelCheckpoint('modelBest-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
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
    #so we reshape our data into their appropriate batches and train our model simulatenously nb_val_samples=len(X_valid_image),
    
    if Continue == False:
        concate_model.compile(loss={'Steer': 'mse', 'Speed': 'categorical_crossentropy'}
                                 ,metrics={'Steer': 'mse', 'Speed':'accuracy'},
            optimizer=Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6))
        history = concate_model.fit_generator(batch_generator(args.data_dir,X_train_image ,X_train_Sequence,Y_train_steer,Y_train_speed, args.batch_size, True,args.samples_per_epoch),
                            args.samples_per_epoch,
                            args.nb_epoch,
                            max_q_size=10,
                            validation_data=batch_generator(args.data_dir, X_valid_image,X_valid_Sequence ,Y_valid_steer ,Y_valid_speed, args.batch_size, False,args.samples_per_epoch),
                            callbacks=[checkpoint,checkpoint2],
                            #verbose = 1,nb_val_samples=len(X_valid_image))
                            #verbose = 1,validation_steps = args.samples_per_epoch)
                            verbose = 1,validation_steps = args.samples_per_epoch*args.batch_size*args.test_size/args.ValidationBatch_Size)
    else:
        history = concate_model.fit_generator(batch_generator(args.data_dir,X_train_image ,X_train_Sequence,Y_train_steer,Y_train_speed, args.batch_size, True,args.samples_per_epoch),
                            args.samples_per_epoch,
                            args.nb_epoch,
                            validation_data=batch_generator(args.data_dir, X_valid_image,X_valid_Sequence ,Y_valid_steer ,Y_valid_speed, args.batch_size, False,args.samples_per_epoch),
                            callbacks=[checkpoint,checkpoint2],
                            verbose = 1,validation_steps = args.samples_per_epoch*args.batch_size*args.test_size/args.ValidationBatch_Size)
                            #verbose = 1,nb_val_samples=len(X_valid_image))
    #visulize steering angle mean square error over training 
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validate'], loc='upper left')
    plt.show()

    #visulize speed mean square error over training 
    plt.plot(history.history['Steer_mse'])
    plt.plot(history.history['val_Steer_mse'])
    plt.title('Steering angle mean_squared_error')
    plt.ylabel('mean_squared_error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    #visulize training loss over training 
    plt.plot(history.history['Speed_accuracy'])
    plt.plot(history.history['val_Speed_accuracy'])
    plt.title('Speed accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    plt.plot(history.history['Speed_loss'])
    plt.plot(history.history['val_Speed_loss'])
    plt.title('Speed loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    plt.plot(history.history['Steer_loss'])
    plt.plot(history.history['val_Steer_loss'])
    plt.title('Steer loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    # Plot training & validation loss values

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def Compute_Timing(args):
    ih = np.zeros((1,227,227,3))
    speed_sequence = np.zeros((1,10,1))
    steer_before ,speed_before , speed_final , steer_final , Total_Model = build_model_speed(args , True)
    now = time.time()
    x  = steer_before.predict(ih)
    #train_model(Total_Model, args, *data)
    #Total_Model.save("model.h5")
    print("delay of steer before concatenate = "+str("{0:.2f}".format(round((time.time()-now),2)))+ "Seconds")  

    now = time.time()
    x  = speed_before.predict(speed_sequence)
    #train_model(Total_Model, args, *data)
    #Total_Model.save("model.h5")
    print("delay of speed before concatenate = "+str("{0:.2f}".format(round((time.time()-now),2)))+ "Seconds")  

    now = time.time()
    x  = speed_final.predict([ih , speed_sequence])
    #train_model(Total_Model, args, *data)
    #Total_Model.save("model.h5")
    print("delay of speed after concatenate = "+str("{0:.2f}".format(round((time.time()-now),2)))+ "Seconds") 

    now = time.time()
    x  = steer_final.predict(ih)
    #train_model(Total_Model, args, *data)
    #Total_Model.save("model.h5")
    print("delay of steer after concatenate = "+str("{0:.2f}".format(round((time.time()-now),2)))+ "Seconds") 

    now = time.time()
    x  = Total_Model.predict([ih , speed_sequence])
    #train_model(Total_Model, args, *data)
    #Total_Model.save("model.h5")
    print("delay of Total_Model predication = "+str("{0:.2f}".format(round((time.time()-now),2)))+ "Seconds")   

def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='D:/Graduation project/CARLA/PythonAPI/examples/speedClassificationModel/Scaled/')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=20)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=1000) #150
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=32) #128
    parser.add_argument('-v', help='batch size',            dest='ValidationBatch_Size',type=int,   default=32) #128
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='false')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1e-4)
    parser.add_argument('-c', help='continue leanring',     dest='continue_learning', type=int,   default=0)
    parser.add_argument('-z', help='compute timing',        dest='Compute_Timingg',    type=int,   default=0)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)
    #Compute_Timing(args)
    #load data
    data = load_data(args)
    
    #build model
    #Total_Model  = build_model_speed(args )        #Steer_Model
    if args.Compute_Timingg:
        Compute_Timing(args)
    elif args.continue_learning:

        Total_Model = load_model('model-014.h5')
        train_model(Total_Model, args,*data , Continue = True)
        Total_Model.save("model.h5")
        print("Done")
    else:
        Total_Model  = build_model_speed(args ) 
        train_model(Total_Model, args,*data , Continue = False) 
        Total_Model.save("model.h5")
        print("Done")
    #train model on data, it saves as model.h5 
    

    #Total_Model.evaluate(batch_generator(args.data_dir,data[1] ,data[5],data[3],data[7], args.batch_size, False))
    #Total_Model.evaluate(batch_generator(args.data_dir,data[1] ,data[5],data[3],data[7], args.batch_size , False), 
    #    verbose=1)
if __name__ == '__main__':
    main()

