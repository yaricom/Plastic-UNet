# The model training

import sys
import os
import pickle

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
from eval import eval_net

def train(net,
          params):
    """
    Do network training
    Arguments:
        net:    The network to be trained
        params: The hyper parameters to use
    """


    # Get train images and masks
    print('Getting train images and masks from dataset ')
    sys.stdout.flush()
    with h5py.File(params['dataset_file'], 'r') as f:
        X = f['train/images'][()]
        y = f['train/masks'][()]

    print('Done!')

    # Split dataset into validation and train data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params['val_ratio'], random_state=42)

    print("Train samples count: %d, validation: %d" % (X_train.shape[0], X_val.shape[0]))

    #
    # Check if training data looks all right
    #
    #if params['debug']:
    #    plot_train_check(X_train, y_train)

    # transpose HWC to CHW image data format accepted by Torch
    X_train = list(map(hwc_to_chw, X_train))
    X_val = list(map(hwc_to_chw, X_val))
    y_train = list(map(hwc_to_chw, y_train))
    y_val = list(map(hwc_to_chw, y_val))

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

    for epoch in range(params['epochs']):
        if params['debug']:
            print('Starting epoch %d/%d.' % (epoch + 1, params['epochs']))

        net.train()

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

            print(loss_num)

            # Compute the gradients
            loss.backward()
            optimizer.step()
            scheduler.step()


        epoch_loss = np.mean(all_losses[-samples_count])
        loss_between_saves += epoch_loss
        if params['debug']:
            print('Epoch finished! Loss: %f' % (epoch_loss))

        #
        # Perform validation
        #
        if (epoch + 1) % params['val_every'] == 0:
            val_acc, val_loss = eval_net(net, X_val, y_val, params['device'], nn.BCELoss())

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
        if (epoch + 1) % params['save_every'] == 0 or (epoch + 1) == epochs:
            print("Saving checkpoint files for epoch:", epoch)

            epochs_since_last_cp = epoch - last_save_epoch # epoch starts from zero
            last_save_epoch = epoch

            if epochs_since_last_cp == 0:
                epochs_since_last_cp = 1
            print("Average loss over the last %d epochs: %f" % \
                    (epochs_since_last_cp, loss_between_saves/epochs_since_last_cp))
            if epoch > 100:
                loss_last_100 = np.mean(all_losses[-samples_count * 100])
                print("Average loss over the last 100 epochs: ", losslast100)

            loss_between_saves = 0.0
            # Save trained data, network parameters and losses
            local_preffix = params['out_dir'] + '/train_data'
            if (epoch + 1) % 50000 == 0:
                local_preffix = local_preffix + "_"+str(epoch + 1)
            with h5py.File(local_preffix + ".hdf5", 'w') as f:
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

def start_train(dataset_file,
                out_dir,
                model,
                load=False,
                gpu=True,
                epochs=5,
                lr=0.1,
                val_ratio=0.05,
                val_every=50,
                save_every=100,
                gamma=0.666,
                steplr=1e6):
    """
    Starts network training
    Arguments:
        dataset_file:   The dataset file to get input data from
        out_dir:        The output directory to store execution results
        model:          The file with network model if needed to load network state before training
        load:           The flag to indicate whether to load network state before
        gpu:            The flag to indicate whether to use GPU
        epochs:         The number of training epochs
        val_ratio:      The ratio of training data to be used for validation
        val_every:      Indicates number of epochs between validation
        save_every:     The number of epoch to execute per results saving
        gamma:          The annealing factor of learning rate decay for Adam
        steplr:         How often should we change the learning rate
    """
    # Create torch device for tensor operations
    device = None
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Put parameters into dictionary
    params = {"dataset_file":dataset_file,
              "out_dir":out_dir,
              "device":device,
              "epochs":epochs,
              "lr":lr,
              "val_ratio":val_ratio,
              "val_every":val_every,
              "save_every":save_every,
              "gamma":gamma,
              "steplr":steplr,
              "im_width":128,
              "im_height":128,
              "im_chan":3,
              "debug":True}

    # Create network structure
    net = UNetp(n_channels=params['im_chan'], n_classes=1, device=device)

    if load:
        net.load_state_dict(torch.load(model))
        print('Model loaded from %s' % (model))

    # do network training
    try:
        train(net, params)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), args.out_dir + '/INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

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
    parser.add_option('--model', '-m', default='MODEL.pth',
                        help="Specify the file in which is stored the model")

    parser.add_option('--save_every', dest='save_every', default=100, type='int',
                      help='save results per specified number of epochs')

    parser.add_option('--validate_every', dest='validate_every', default=50, type='int',
                      help='validate model per specified number of epochs')

    parser.add_option('-i', '--data', dest='data_file', type='string',
                      help='the path to the dataset file with input data')
    parser.add_option('-o', '--out', dest='out_dir', type='string',
                      help='the path to the directory for results ouput')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = parse_args()

    # Check if output directory exists
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # start network training
    start_train(dataset_file=args.data_file,
                out_dir=args.out_dir,
                model=args.model,
                load=args.load,
                gpu=args.gpu,
                epochs=args.epochs,
                lr=args.lr,
                save_every=args.save_every,
                val_every=args.validate_every)
