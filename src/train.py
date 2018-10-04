# The model training

import sys
import os

import warnings
from optparse import OptionParser

import h5py
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from unet import UNetp
from utils import plot_train_check
from utils import hwc_to_chw

# Set some parameters
im_width = 128
im_height = 128
im_chan = 3
debug = True

def train(net,
          dataset_file,
          out_dir,
          device,
          epochs=5,
          lr=0.1,
          val_ratio=0.05,
          val_every=5,
          save_every=5000,
          gamma=0.666,
          steplr=1e6):
    """
    Starts network training
    Arguments:
        net: The network to be trained
        dataset_file: The dataset file to get input data from
        out_dir: The output directory to store execution results
        device: The Torch device to execute Tensors on
        epochs: The number of training epochs
        val_ratio: The ratio of training data to be used for validation
        val_every: Indicates number of epochs between validation
        save_every: The number of epoch to execute per results saving
        gamma: The annealing factor of learning rate decay for Adam
        steplr: How often should we change the learning rate
    """
    # Get train images and masks
    print('Getting train images and masks from dataset ')
    sys.stdout.flush()
    with h5py.File(dataset_file, 'r') as f:
        X = f['train/images'][()]
        y = f['train/masks'][()]

    print('Done!')

    # Split dataset into validation and train data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42)

    print("Train samples count: %d, validation: %d" % (X_train.shape[0], X_val.shape[0]))

    #
    # Check if training data looks all right
    #
    #if debug:
    #    plot_train_check(X_train, y_train)

    # transpose HWC to CHW image data format accepted by Torch
    X_train = list(map(hwc_to_chw, X_train))
    X_val = list(map(hwc_to_chw, X_val))
    y_train = list(map(hwc_to_chw, y_train))
    y_val = list(map(hwc_to_chw, y_val))

    #
    # Initialize optimizer
    #
    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=steplr)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        if debug:
            print('Starting epoch %d/%d.' % (epoch + 1, epochs))

        # Initialize Hebbian with zero values for new epoch
        hebb = net.initialZeroHebb()

        epoch_loss = 0

        net.train()
        # Enumerate over samples and do train
        for img, mask in zip(X_train, y_train):
            t_img = torch.from_numpy(np.array([img.astype(np.float32)])).to(device)
            y_target = torch.from_numpy(mask.astype(np.float32)).to(device)

            # Starting each sample, we detach the Hebbian state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            y_pred, hebb = net(t_img, Variable(hebb))

            y_pred_flat = y_pred.view(-1)
            y_target_flat = y_target.view(-1)

            # compute loss
            loss = criterion(y_pred_flat, Variable(y_target_flat, requires_grad=False))
            epoch_loss += loss.item()

            print("Loss: %s, epoch loss: %f" % (loss.item(), epoch_loss))

            # Compute the gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if debug:
            print('Epoch finished ! Loss: {}'.format(epoch_loss / len(X_train)))



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
              device=args.device,
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
