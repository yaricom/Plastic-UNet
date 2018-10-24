# The model training

import sys
import os
import pickle
import time

import warnings
from optparse import OptionParser

import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from unet import UNetp

from utils import plot_train_check
from utils import load_image

from utils import plot_coverage
from utils import plot_depth

from eval import eval_net

def train(net, X_train, X_val, y_train, y_val, params):
    """
    Do network training
    Arguments:
        net:        The network to be trained
        X_train:    The training samples
        X_val:      The validation samples
        y_train:    The training labels
        y_val:      The validation labels
        params:     The hyper parameters to use
    """

    print("Train samples shape:", X_train.shape)
    print("Train labels shape:", y_train.shape)
    print("Validation samples shape:", X_val.shape)
    print("Validation labels shape:", y_val.shape)
    print(params)

    #
    # The data accumulators
    #
    # The loss values collected over each execution
    all_losses = []

    # The data values collected per epoch when validation happens
    val_train_losses = []
    val_test_losses = []
    val_accuracies = []

    samples_count = len(X_train)
    loss_between_saves = 0.0
    last_save_epoch = 0

    #
    # Initialize optimizer
    #
    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'],
                                                step_size=params['steplr'])

    criterion = nn.BCELoss()

    if params['stop_time'] > 0:
        print("Training started at: %d sec and set to stop at: %d sec" %
                (time.time(), params['stop_time']))

    for epoch in range(params['epochs']):
        if params['debug']:
            print('Starting epoch %d/%d.' % (epoch + 1, params['epochs']))

        net.train()

        # Store epoch start time
        epoch_start_time = time.time()

        # Initialize Hebbian with zero values for new epoch
        hebb = net.initialZeroHebb()

        # Enumerate over samples and do train
        for img, mask in zip(X_train, y_train):
            optimizer.zero_grad()

            t_img = torch.from_numpy(np.array([img.astype(np.float32)])).to(params['device'])
            y_target = torch.from_numpy(mask.astype(np.float32)).to(params['device'])

            # Starting each sample, we detach the Hebbian state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            y_pred, hebb = net(Variable(t_img, requires_grad=False), Variable(hebb, requires_grad=False))

            y_pred_flat = y_pred.view(-1)
            y_target_flat = y_target.view(-1)

            # compute loss
            loss = criterion(y_pred_flat, Variable(y_target_flat, requires_grad=False))
            loss_num = loss.item()
            all_losses.append(loss_num)

            # Compute the gradients
            loss.backward()
            optimizer.step()
            scheduler.step()


        epoch_loss = np.mean(all_losses[-samples_count])
        loss_between_saves += epoch_loss

        epoch_time = time.time() - epoch_start_time
        next_epoch_finish_time = epoch_time + time.time()
        # check if need to force stop training due to time limits
        terminate_training = (params['stop_time'] > 0 and next_epoch_finish_time >= params['stop_time'])

        if params['debug']:
            print('Epoch finished! Loss: %f, time spent: %d, terminate due to time limits: %s' %
                    (epoch_loss, epoch_time, terminate_training))

        #
        # Perform validation
        #
        if (epoch + 1) % params['val_every'] == 0:
            val_acc, val_loss = eval_net(net,
                                         X_val=X_val,
                                         y_val=y_val,
                                         device=params['device'],
                                         criterion=nn.BCELoss(),
                                         debug=params['debug'])

            # store loss values
            val_train_losses.append(epoch_loss)
            val_test_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if params['debug']:
                print('Validation accuracy: %f, loss: %f' % (val_acc, val_loss))
                print ("Eta:", net.eta.data.cpu().numpy())
                sys.stdout.flush()


        #
        # Save checkpoint if appropriate
        #
        if (epoch + 1) % params['save_every'] == 0 or (epoch + 1) == params['epochs'] or terminate_training:
            if params['debug']:
                print("Saving checkpoint files for epoch:", epoch)

            epochs_since_last_cp = epoch - last_save_epoch # epoch starts from zero
            last_save_epoch = epoch

            if epochs_since_last_cp == 0:
                epochs_since_last_cp = 1

            if params['debug']:
                print("Average loss over the last %d epochs: %f" % \
                    (epochs_since_last_cp, loss_between_saves/epochs_since_last_cp))

            if epoch > 100:
                loss_last_100 = np.mean(all_losses[-samples_count * 100])
                if params['debug']:
                    print("Average loss over the last 100 epochs: ", loss_last_100)

            loss_between_saves = 0.0
            # Save trained data, network parameters and losses
            local_preffix = params['out_dir'] + '/train'
            if (epoch + 1) % params['rollout'] == 0 and not terminate_training:
                local_preffix = local_preffix + "_"+str(epoch + 1)

            with h5py.File(local_preffix + "_data.hdf5", 'w') as f:
                f.create_dataset("net/w", data=net.w.data.cpu().numpy(),
                                 compression="gzip", shuffle=True, fletcher32=True)
                f.create_dataset("net/alpha", data=net.alpha.data.cpu().numpy(),
                                 compression="gzip", shuffle=True, fletcher32=True)
                f.create_dataset("net/eta", data=net.eta.data.cpu().numpy(),
                                 compression="gzip", shuffle=True, fletcher32=True)

                f.create_dataset("train/all_losses", data=all_losses,
                                 compression="gzip", shuffle=True, fletcher32=True)

                f.create_dataset("validation/train_losses", data=val_train_losses,
                                 compression="gzip", shuffle=True, fletcher32=True)
                f.create_dataset("validation/test_losses", data=val_test_losses,
                                 compression="gzip", shuffle=True, fletcher32=True)
                f.create_dataset("validation/accuracies", data=val_accuracies,
                                 compression="gzip", shuffle=True, fletcher32=True)

                f.flush()

            # Save training paprameters
            with open(local_preffix+'_parameters.dat', 'wb') as fo:
                pickle.dump(params, fo)

            # Save network state dictionary
            torch.save(net.state_dict(), local_preffix + "_net.pth")

        # Terminate training loop due to time limits
        if terminate_training:
            print("Training terminated due to the time limits!")
            print("Current epoch %d, train loss: %s" % (epoch, epoch_loss))
            print("Stop time limit: %d, estimated time of next epoch end: %d" %
                    (params['stop_time'], next_epoch_finish_time))
            break
"""
def find_best_threshold(net,
                        data_dir,
                        img_width=128,
                        img_height = 128,
                        img_chan=3,
                        gpu=True,
                        visualize=False,
                        save_results=False,
                        debug=False):

    Finds best threshold value for IoU metric against validation data.
    Arguments:
        net: The trained network to use for inference
        data_dir: The directory with data samples
        img_width:      The width of the resized image
        img_height:     The height of the resized image
        visualize:      The flag to indicate whether to plot results
        save_masks:     The flag to indicate whether to save the results
        img_chan:       The number of channels in input plot_image
        debug:          The flag to indicate whether to show debug information
    """



def start_train(x_train, x_valid, y_train, y_valid,
                out_dir,
                model,
                img_width,
                img_height,
                img_chan,
                max_train_time=-1,
                load=False,
                gpu=True,
                epochs=5,
                lr=3e-5,
                val_ratio=0.05,
                val_every=50,
                save_every=100,
                gamma=0.666,
                steplr=1e6,
                rollout=50000,
                prule="hebb",
                debug=False):
    """
    Starts network training
    Arguments:
        x_train:        The training samples
        x_valid:        The validation samples
        y_train:        The training labels
        y_valid:        The validation labels
        out_dir:        The output directory to store execution results
        model:          The file with network model if needed to load network state before training
        load:           The flag to indicate whether to load network state before
        max_train_time: The maximal time in seconds to spend on training
        gpu:            The flag to indicate whether to use GPU
        epochs:         The number of training epochs
        val_ratio:      The ratio of training data to be used for validation
        val_every:      Indicates number of epochs between validation
        save_every:     The number of epoch to execute per results saving
        gamma:          The annealing factor of learning rate decay for Adam
        steplr:         How often should we change the learning rate
        rollout:        The number of epochs to pass before file names rollout
        prule:          The plastic rule to use when training
    Returns:
        The trained network
    """
    # Create torch device for tensor operations
    device = None
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if max_train_time > 0:
        stop_time = time.time() + max_train_time
    else:
        stop_time = -1
    # Put parameters into dictionary
    params = {"out_dir":out_dir,
              "device":device,
              "epochs":epochs,
              "stop_time":stop_time,
              "lr":lr,
              "val_ratio":val_ratio,
              "val_every":val_every,
              "save_every":save_every,
              "rollout":rollout,
              "gamma":gamma,
              "steplr":steplr,
              "prule":prule,
              "im_width":img_width,
              "im_height":img_height,
              "im_chan":img_chan,
              "debug":debug}

    # Create network structure
    net = UNetp(n_channels=params['im_chan'],
                n_classes=1,
                nbf=img_width, 
                batch_norm=False,
                bilinear_upsample=False,
                device=device,
                rule=prule)

    if load:
        net.load_state_dict(torch.load(model))
        print('Model loaded from %s' % (model))

    # do network training
    try:
        train(net=net,
              X_train=x_train,
              X_val=x_valid,
              y_train=y_train,
              y_val=y_valid,
              params=params)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), args.out_dir + '/INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    return net

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i

def load_train_dataset( data_dir,
                        img_width,
                        img_height,
                        img_chan,
                        val_ratio=0.2,
                        debug=False):
    """
    Loads train data set from provided directory and split it onto train and validation.
    Arguments:
        data_dir:   The data directory to load training data
        img_width:  The target image width
        img_height: The target image height
        img_chan:   The target image channels
        val_ratio:  The ratio to split train/test data
    Returns:
        The preprocessed dataset with data samples and target labels for training
        and validation.
    """
    # Create data DataFrame
    train_df = pd.read_csv(data_dir + '/train.csv', index_col='id', usecols=[0])
    depths_df = pd.read_csv(data_dir + '/depths.csv', index_col='id')
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    # Load image data
    train_df["images"] = [np.array(load_image('{}/train/images/{}.png'.format(data_dir, idx), (img_height, img_width))) for idx in train_df.index]
    train_df["masks"] = [np.array(load_image('{}/train/masks/{}.png'.format(data_dir, idx), (img_height, img_width))) / 65535 for idx in train_df.index]

    # Calculate and add salt coverage data
    train_df["coverage"] = train_df.masks.map(np.sum) / (img_height * img_width)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

    # Plots salt coverage and depth distributions
    if debug:
        print(train_df.masks[10])
        plot_coverage(train_df)
        plot_depth(train_df, test_df)

    # Create train/validation split stratified by salt coverage
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.tolist()).reshape(-1, img_chan, img_height, img_width),
        np.array(train_df.masks.tolist()).reshape(-1, 1, img_height, img_width),
        train_df.coverage.values,
        train_df.z.values,
        test_size=val_ratio, stratify=train_df.coverage_class, random_state=42)


    return x_train, x_valid, y_train, y_valid

def parse_args():
    """
    Parses command line arguments
    """
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-l', '--learning-rate', dest='lr', default=3e-5,
                      type='float', help='learning rate')
    parser.add_option('-s', '--step-lr', dest='steplr', default=1e6,
                      type='float', help='the learning rate annealing step')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')

    parser.add_option('--prule', '-p', default='hebb',
                        help="the plastic rule to use when training")

    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('--model', '-m', default='MODEL.pth',
                        help="Specify the file in which is stored the model")

    parser.add_option('--max-train-time', dest='max_train_time', default=-1, type='int',
                      help='used to specify max training time limit in seconds [default: -1 which means no limits]')
    parser.add_option('--save_every', dest='save_every', default=100, type='int',
                      help='save results per specified number of epochs')
    parser.add_option('--validate_every', dest='validate_every', default=50, type='int',
                      help='validate model per specified number of epochs')
    parser.add_option('--rollout_every', dest='rollout_every', default=50000, type='int',
                      help='rollout output files every # of epochs')

    parser.add_option('-d', '--data', dest='data_dir', type='string',
                      help='the directory with input data')
    parser.add_option('-i', '--dataset', dest='dataset_file', type='string',
                      help='the path to the dataset file with input data')
    parser.add_option('-o', '--out', dest='out_dir', type='string',
                      help='the path to the directory for results ouput')

    parser.add_option('-v', '--debug', action='store_true', dest='debug',
                      default=False, help='show debug information')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = parse_args()

    # Check if output directory exists
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    print(args)

    # Set values
    t_img_width=101
    t_img_height=101
    t_img_chan=1

    # Get train images and masks
    if args.data_dir != None:
        print('Getting train images and masks from data directory %s' % args.data_dir)
        sys.stdout.flush()
        x_train, x_valid, y_train, y_valid = load_train_dataset(data_dir=args.data_dir,
                                                                img_width=t_img_width,
                                                                img_height=t_img_height,
                                                                img_chan=t_img_chan,
                                                                debug=args.debug)
        print('Done!')
    else:
        raise ValueError("The input data directory or dataset file not specified")

    # start network training
    start_train(x_train, x_valid, y_train, y_valid,
                out_dir=args.out_dir,
                model=args.model,
                load=args.load,
                gpu=args.gpu,
                epochs=args.epochs,
                lr=args.lr,
                steplr=args.steplr,
                max_train_time=args.max_train_time,
                save_every=args.save_every,
                val_every=args.validate_every,
                rollout=args.rollout_every,
                prule=args.prule,
                img_width=t_img_width,
                img_height=t_img_height,
                img_chan=t_img_chan,
                debug=args.debug)
