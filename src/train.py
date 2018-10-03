# The model training

import sys
import os

import warnings
from optparse import OptionParser

import h5py

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from unet import UNetp
from utils import plot_train_check

# Set some parameters
im_width = 128
im_height = 128
im_chan = 3

def train(net,
          dataset_file,
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
        dataset_file: The dataset file to get input data from
        out_dir: The output directory to store execution results
        epochs: The number of training epochs
        val_ratio: The ratio of training data to be used for validation
        save_every: The number of epoch to execute per results saving
        gamma: The annealing factor of learning rate decay for Adam
        steplr: How often should we change the learning rate
    """
    # Get train images and masks
    print('Getting train images and masks from dataset ')
    sys.stdout.flush()
    with h5py.File(dataset_file, 'r') as f:
        X_train = f['train/images'][()]
        Y_train = f['train/masks'][()]

    print('Done!')

    #
    # Check if training data looks all right
    #
    plot_train_check(X_train, Y_train)

    #
    # Initialize optimizer
    #
    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=steplr)

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
    parser.add_option('-i', '--data', dest='data_file', type='string',
                      help='the path to the dataset file with input data')
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
              dataset_file=args.data_file,
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
