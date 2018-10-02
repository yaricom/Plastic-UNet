# The model training

import sys
import os
import random
import warnings
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from unet import UNetp

# Set some parameters
im_width = 128
im_height = 128
im_chan = 3

def train(net,
          data_dir,
          out_dir,
          epochs=5,
          lr=0.1,
          val_ratio=0.05,
          save_every=5000,
          gamma=0.666,
          steplr=1e6):
    """
    Starts network training
    Arguments:
        net: The network to be trained
        data_dir: The directory to get input data from
        out_dir: The output directory to store execution results
        epochs: The number of training epochs
        val_ratio: The ratio of training data to be used for validation
        save_every: The number of epoch to execute per results saving
        gamma: The annealing factor of learning rate decay for Adam
        steplr: How often should we change the learning rate
    """
    train_ids = next(os.walk(data_dir + "/images"))[2]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float64)
    Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in enumerate(train_ids):
        path = data_dir
        x = load_image(path + '/images/' + id_, (128, 128, im_chan))
        X_train[n] = x
        mask = load_image(path + '/masks/' + id_, (128, 128, 1))
        Y_train[n] = mask

    print('Done!')

    #
    # Check if training data looks all right
    #
    ix = random.randint(0, len(train_ids))
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(X_train[ix])
    a.set_title('Image')
    a = fig.add_subplot(1, 2, 2)
    tmp = np.squeeze(Y_train[ix]).astype(np.float32)
    plt.imshow(np.dstack((tmp,tmp,tmp)))
    a.set_title('Mask')
    plt.show()

    #
    # Initialize optimizer
    #
    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=steplr)


def load_image(path, output_shape):
    img = imread(path)
    x = resize(img, output_shape, mode='constant')
    return x

def parse_args():
    """
    Parses command line arguments
    """
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-l', '--learning-rate', dest='lr', default=3e-5,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('--save_every', dest='save_every', default=5, type='int',
                      help='save results per specified number of epochs')
    parser.add_option('-i', '--data', dest='data_dir', type='string',
                      help='the path to the directory with input data')
    parser.add_option('-o', '--out', dest='out_dir', type='string',
                      help='the path to the directory for results ouput')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = parse_args()

    # Create torch device for tensor operations
    args.device = None
    if args.gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # Create network structure
    net = UNetp(n_channels=im_chan, n_classes=1, device=args.device)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    try:
        train(net=net,
              data_dir=args.data_dir,
              out_dir=args.out_dir,
              epochs=args.epochs,
              lr=args.lr,
              save_every=args.save_every)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
