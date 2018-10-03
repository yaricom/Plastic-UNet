# The image utilities
import os
import random
import sys

from optparse import OptionParser

import h5py

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize

def load_image(path, output_shape):
    """
    Loads image under specified path and resize it to conform given output shape
    """
    img = imread(path)
    x = resize(img, output_shape, mode='constant')
    return x

def create_hdf5_data_set(data_dir,
                         out_file="dataset.hdf5",
                         im_width=128,
                         im_height = 128,
                         im_chan = 3):
    """
    Creates data set as HDF5 file
    Arguments:
        data_dir: The directory to look for data files
        out_file: The name of the dataset file
        im_width: The expected width of images
        im_height: The expected height of images
        im_chan: The expected number of channels
    """
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()

    # Get and resize train images and masks
    train_ids = next(os.walk(data_dir + "/train/images"))[2]
    X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float64)
    Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
    for n, id_ in enumerate(train_ids):
        path = data_dir
        x = load_image(path + '/train/images/' + id_, (im_width, im_height, im_chan))
        X_train[n] = x
        mask = load_image(path + '/train/masks/' + id_, (im_width, im_height, 1))
        Y_train[n] = mask

    print('Done!')

    #
    # Check if training data looks all right
    #
    plot_train_check(X_train, Y_train)

    print('Getting and resizing test images... ')
    test_ids = next(os.walk(data_dir + "/test/images"))[2]
    X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.float64)
    for n, id_ in enumerate(test_ids):
        x = load_image(data_dir + '/test/images/' + id_, (im_width, im_height, im_chan))
        X_test[n] = x

    print('Done!')

    #
    # Check if test data looks all right
    #
    plot_test_ckeck(X_test)

    out_path = data_dir + "/" + out_file
    print('Creation of HDF5 dataset file at: %s' % out_path)
    with h5py.File(out_path,'w') as f:
        f.create_dataset("train/images", data=X_train, compression="gzip", shuffle=True, fletcher32=True)
        f.create_dataset("train/masks", data=Y_train, compression="gzip", shuffle=True, fletcher32=True)

        f.create_dataset("test/images", data=X_test, compression="gzip", shuffle=True, fletcher32=True)

        f.flush()

    print('Done!')

def plot_train_check(X_train, Y_train):
    ix = random.randint(0, len(X_train) - 1)
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(X_train[ix])
    a.set_title('Image')
    a = fig.add_subplot(1, 2, 2)
    tmp = np.squeeze(Y_train[ix]).astype(np.float32)
    plt.imshow(np.dstack((tmp,tmp,tmp)))
    a.set_title('Mask')
    plt.show()

def plot_test_ckeck(X_test):
    ix = random.randint(0, len(X_test) - 1)
    plt.imshow(X_test[ix])
    plt.show()

def parse_args():
    """
    Parses command line arguments
    """
    parser = OptionParser()
    parser.add_option('--action', dest='action', default='create_dataset', type='string',
                      help='the action to be performed')
    parser.add_option('-i', '--data', dest='data_dir', type='string',
                      help='the directory with input data')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = parse_args()

    if args.action == 'create_dataset':
        create_hdf5_data_set(data_dir=args.data_dir)
    else:
        raise ValueError("Usuported action requested: %s" % args.action)
