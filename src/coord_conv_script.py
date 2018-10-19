# The TF Keras based one script solution with CoordConvolution layer introduced
import os
import sys
import random
import warnings
import time

import pandas as pd
import numpy as np

import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend as K
from keras.engine.topology import Layer

import tensorflow as tf
from tensorflow.python.layers import base

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

############################################################
# The callback to stop training if time exceeds alotted ones
############################################################
class TimedStopping(Callback):

    def __init__(self, max_train_time):
        super(TimedStopping, self).__init__()

        self.max_train_time = max_train_time
        self.stop_time = time.time() + max_train_time
        self.epoch_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        next_epoch_finish_time = epoch_time + time.time()
        terminate_training = (self.stop_time > 0 and next_epoch_finish_time >= self.stop_time)
        if terminate_training:
            print("Training terminated due to the time limits............")
            self.model.stop_training = True

############################################################
# The coordinate convolution implementation
############################################################
class AddCoords(Layer):
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        """Add coords to a tensor"""
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        super(AddCoords, self).__init__()

    def call(self, input_tensor):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        batch_size_tensor = tf.shape(input_tensor)[0]
        xx_ones = tf.ones([batch_size_tensor, self.x_dim], dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0), [batch_size_tensor, 1])
        xx_range = tf.expand_dims(xx_range, 1)
        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, -1)

        yy_ones = tf.ones([batch_size_tensor, self.y_dim], dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0), [batch_size_tensor, 1])
        yy_range = tf.expand_dims(yy_range, -1)
        yy_channel = tf.matmul(yy_range, yy_ones)
        yy_channel = tf.expand_dims(yy_channel, -1)
        xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)
        xx_channel = xx_channel*2 - 1
        yy_channel = yy_channel*2 - 1

        ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)
        if self.with_r:
            rr = tf.sqrt( tf.square(xx_channel-0.5) + tf.square(yy_channel-0.5) )
            ret = tf.concat([ret, rr], axis=-1)
        return ret

    def compute_output_shape(self, input_shape):
        channels = 3
        if self.with_r:
            channels = 4
        return (input_shape[0], input_shape[1], input_shape[2], channels)

class CoordConv(Layer):

    def __init__(self, x_dim, y_dim, with_r, *args, **kwargs):
        """CoordConv layer as in the paper."""
        self.addcoords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
        self.conv = Conv2D(**kwargs)
        super(CoordConv, self).__init__()

    def call(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.conv.filters)

    def get_config(self):
        config = {'x_dim':self.addcoords.x_dim,
                  'y_dim':self.addcoords.y_dim,
                  'with_r':self.addcoords.with_r
                  }
        conv_config = self.conv.get_config()
        return dict(list(conv_config.items()) + list(config.items()))

############################################################
# Define IoU metric
############################################################
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

############################################################
# The CNN U-Net pyramide build with coordinated convolution layers
############################################################

def construct_model(im_height, im_width, im_chan):
    with_r = True

    inputs = Input((im_height, im_width, im_chan))
    s = Lambda(lambda x: x / 255) (inputs)

    print("Inputs shape:", inputs)

    cc1 = CoordConv(im_height, im_width, with_r, filters=8, kernel_size=(1, 1), activation='relu', padding='same') (s)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (cc1)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    #model.summary()

    return model


############################################################
# Do training with given model
############################################################

def do_training(model, X_train, Y_train, epochs, max_train_time, model_file, verbose=0):
    print("Training started at: %d sec and set to be run for: %d sec" %
            (time.time(), max_train_time))

    earlystopper = EarlyStopping(patience=5, verbose=verbose)
    checkpointer = ModelCheckpoint(model_file, verbose=verbose, save_best_only=True)
    timedstopper = TimedStopping(max_train_time)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=epochs,
                        callbacks=[earlystopper, checkpointer, timedstopper], verbose=verbose)

    print('Traing Complete!')

def start_training(train_ids, epochs, max_train_time, path_train, im_height, im_width, im_chan, model_file, verbose=0):
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ', len(train_ids))
    sys.stdout.flush()
    for n, id_ in enumerate(train_ids):
        path = path_train
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,1]
        x = resize(x, (im_height, im_width, im_chan), mode='constant', preserve_range=True)
        X_train[n] = x
        mask = img_to_array(load_img(path + '/masks/' + id_))[:,:,1]
        Y_train[n] = resize(mask, (im_height, im_width, im_chan), mode='constant', preserve_range=True)

    print('Done!')

    # Do model fitting
    model = construct_model(im_height, im_width, im_chan)
    do_training(model=model,
                X_train=X_train,
                Y_train=Y_train,
                epochs=epochs,
                max_train_time=max_train_time,
                model_file=model_file,
                verbose=verbose)



############################################################
# Starts inference with given model
############################################################
def start_prediction(model_file, test_ids, path_test, im_height, im_width, im_chan, verbose=0):
    X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ', len(test_ids))
    sys.stdout.flush()
    for n, id_ in enumerate(test_ids):
        path = path_test
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,1]
        sizes_test.append([x.shape[0], x.shape[1]])
        x = resize(x, (im_height, im_width, im_chan), mode='constant', preserve_range=True)
        X_test[n] = x

    print('Done!')

    # Loading model and do prediction
    print('Loading model from:', model_file)
    custom_objects={'CoordConv':CoordConv,
                    'mean_iou': mean_iou}
    model = load_model(model_file, custom_objects=custom_objects)

    print('Start prediction!')
    preds_test = model.predict(X_test, verbose=verbose)

    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in tnrange(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    print('Prediction complete! Output images shape:', preds_test_upsampled[0].shape)

    return preds_test_upsampled

############################################################
# RLE encode submission file
############################################################
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

############################################################
# Set some parameters
############################################################
im_width = 128
im_height = 128
im_chan = 1
input_data_dir="../input"
data_dir=input_data_dir + "/tgs-salt-identification-challenge"
path_train = data_dir + '/train/'
path_test = data_dir + '/test/'
model_file_name = 'model-tgs-salt-4.h5'

epochs=500
max_train_time=14500#21000#19600 #180#

do_train = True#False#
do_inference = False#True#

short_run = False#True#

train_ids = next(os.walk(path_train+"images"))[2]
test_ids = next(os.walk(path_test+"images"))[2]

if short_run:
    train_ids = train_ids[:100]
    test_ids = test_ids[:100]

if do_train:
    print("Start training")

    start_training(train_ids=train_ids,
                   epochs=epochs,
                   max_train_time=max_train_time,
                   path_train=path_train,
                   im_height=im_height,
                   im_width=im_width,
                   im_chan=im_chan,
                   model_file=model_file_name,
                   verbose=0)

###################################
# Do prediction
###################################
model_file=input_data_dir + '/coord-conv-model/' + model_file_name
subm_file="submission.csv"

if do_inference:
    print("Starting inference with model:", model_file)

    predicted = start_prediction(model_file=model_file,
                                 test_ids=test_ids,
                                 path_test=path_test,
                                 im_height=im_height,
                                 im_width=im_width,
                                 im_chan=im_chan,
                                 verbose=0)

    # Prepare submission file
    pred_dict = {fn[:-4]:RLenc(np.round(predicted[i])) for i, fn in enumerate(test_ids)}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(subm_file)
