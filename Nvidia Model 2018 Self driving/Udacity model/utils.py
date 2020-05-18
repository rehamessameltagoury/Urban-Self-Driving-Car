import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS =227,227,3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def Append_Sequence(listt , value):
    for i in range(0,9):
        listt[i] = listt[i+1]
    listt[9] = value
    return listt

'''def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    #print ("-------------------------------------------------------------")
    #print (image_file)
    #print ("-------------------------------------------------------------")
    #return Image.open(image_file)
    #return image
    #img=mpimg.read(image_file)
    #return mpimg.imread(os.path.join(data_dir, image_file.strip()))
    #IMAGE =mpimg.imread(image_file.strip())
    #plt.imshow(IMAGE)
'''   #return mpimg.imread(image_file.strip())
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
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """

    choice = np.random.choice(3)
    if choice == 0:
        if steering_angle > 0.8:
            return load_image(data_dir, left) , steering_angle
        else:
            return load_image(data_dir, left) ,steering_angle + 0.2
    elif choice == 1:
        if steering_angle < -0.8:
            return load_image(data_dir, right) , steering_angle
        else:
            return load_image(data_dir, right) ,steering_angle - 0.2
    
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

'''
def augument(data_dir, center, steering_angle, range_x=100, range_y=10):#**
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, steering_angle)#**
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle
'''

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

'''
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center= image_paths[index] #**
            print(center)
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center,steering_angle)#**
            else:
                image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
'''

def batch_generator(data_dir, image_paths, speed_sequences , steering_angles,speeds ,batch_size, is_training,sample_per_epoch):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    speed_seq = np.empty((batch_size,10,1))
    speed     = np.empty((batch_size,3))
    temp = np.zeros((1,10))
    #speed = np.empty(batch_size)
    throttle = 0
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            #print(center)
            steering_angle = steering_angles[index]

            speed_index        = str_to_float(speeds[index])
            speed_seqq       = str_to_float(speed_sequences[index])
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = float("{:.3f}".format((steering_angle+1)))
            temp = np.array(speed_seqq)
            temp = temp.reshape((10,1))

            xx = np.array(speed_index)
            xx = xx.reshape((3))

            speed_seq[i,:,:] = temp
            speed[i,:] = xx
            i += 1
            if i == batch_size:
                break
        #print(speed)
        speed_seq = speed_seq.reshape((batch_size,10,1))
        speed = speed.reshape((batch_size,3))
        #print(speed)
        #print(steers)
        inputs = [ images , speed_seq ] 
        outputs = [speed , steers]
        yield inputs , outputs