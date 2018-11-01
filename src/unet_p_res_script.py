# The One page script with plastic U-Net residual implementation
import os
import sys
import random
import warnings
import pickle
import time
from datetime import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import h5py

from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable
from torch.autograd import no_grad

# Set some values
start_neurons=8

def load_image(path, output_shape):
    """
    Loads image under specified path and resize it to conform given output shape
    if appropriate
    """
    img = imread(path, as_grey=True)
    if img.shape != output_shape:
        img = resize(img, output_shape, mode='constant', preserve_range=True)
    return img

#########################################

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def fast_iou_metric(y_true_in, y_pred_in):
    iou = get_iou_vector(y_true_in, y_pred_in>0.5)
    return iou

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
    intersection = temp1[0]
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

#########################################

def eval_net(net, X_val, y_val, device, criterion, debug=False):
    """
    Perorms network evaluation
    Arguments:
        net:        The network to be evaluated
        X_val:      The data samples for validation
        y_val:      The ground truth data for validation
        device:     The Torch device to use
        criterion:  The loss function to use
        debug:      The flag to indicate if debug info should be displayed
    Returns:
        (accuracy, loss) the tuple with validation accuracy against IoU metric as well as validation loss
    """
    net.eval()
    with torch.no_grad():
        hebb = net.initialZeroHebb()

        val_loss = 0
        total_acc = 0
        for i, d in enumerate(zip(X_val, y_val)):
            t_img = torch.from_numpy(np.array([d[0].astype(np.float32)])).to(device)
            mask_val = torch.from_numpy(d[1].astype(np.float32)).to(device)

            # We do not learn plasticity within validation
            mask_pred, _ = net(Variable(t_img, requires_grad=False), Variable(hebb, requires_grad=False))

            y_pred_flat = mask_pred.view(-1)
            y_target_flat = mask_val.view(-1)

            # The validation loss
            loss = criterion(y_pred_flat, y_target_flat)
            val_loss += loss.item()

            # The validation accuracy
            acc = fast_iou_metric(y_pred_in=y_pred_flat.cpu().numpy(), y_true_in=y_target_flat.cpu().numpy())
            total_acc += acc

    return (total_acc / (i + 1), val_loss / (i + 1))

#########################################

class UNetpRes(nn.Module):
    def __init__(self, n_channels, n_classes, device, neurons=16, dropout_ratio=0.5, alfa_type='free', rule='hebb', nbf=128, batch_norm=False, bilinear_upsample=False):
        """
        Creates new U-Net network with plastic learning rule implemented
        Arguments:
            n_channels: The number of input n_channels
            n_classes:  The number of ouput classes to be learned
            device:     The torch device to execute tensor operations
            neurons:    The # of neurons for the first leayer
            alfa_type:  The plasticity coefficient ['free', 'yoked'] (if the latter, alpha is a single scalar learned parameter, shared across all connection)
            rule:       The name of plasticity rule to apply ['hebb', 'oja'] (The Oja rule can maintain stable weight values indefinitely in the absence of stimulation, thus allowing stable long-term memories, while still preventing runaway divergences)
            nbf:        The number of features in plasticity rule vector (width * height)
        """
        super(UNetpRes, self).__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.nbf = nbf # the number of features to be used for plastic rule learning
        self.torch_dev = device
        self.alfa_type = alfa_type
        self.rule = rule

        # The plastic rule paprameters to be learned
        self.w =  torch.nn.Parameter((.01 * torch.randn(self.nbf, self.nbf, device=self.torch_dev)), requires_grad=True) # Fixed weights
        self.alpha =  torch.nn.Parameter((.01 * torch.rand(self.nbf, self.nbf, device=self.torch_dev)), requires_grad=True) # Plasticity coeffs.
        self.eta = torch.nn.Parameter((.01 * torch.ones(1, device=self.torch_dev)), requires_grad=True)  # The “learning rate” of plasticity (the same for all connections)

        # The DOWN network structure
        # 101 -> 50
        self.conv1 = down(n_channels, neurons, batch_norm=batch_norm)
        self.pool1 = pool_drop(dropout_ratio=dropout_ratio/2)
        # 50 -> 25
        self.conv2 = down(neurons, neurons * 2, batch_norm=batch_norm)
        self.pool2 = pool_drop(dropout_ratio=dropout_ratio)
        # 25 -> 12
        self.conv3 = down(neurons * 2, neurons * 4, batch_norm=batch_norm)
        self.pool3 = pool_drop(dropout_ratio=dropout_ratio)
        # 12 -> 6
        self.conv4 = down(neurons * 4, neurons * 8, batch_norm=batch_norm)
        self.pool4 = pool_drop(dropout_ratio=dropout_ratio)

        # Middle
        self.mid = middle(neurons * 8, neurons * 16, batch_norm=batch_norm)

        # The UP network structure
        # 6 -> 12
        self.uconv4 = up(neurons * 16, neurons * 8, dropout_ratio=dropout_ratio, batch_norm=batch_norm)
        # 12 -> 25
        self.uconv3 = up(neurons * 8, neurons * 4, dropout_ratio=dropout_ratio, batch_norm=batch_norm)
        # 25 -> 50
        self.uconv2 = up(neurons * 4, neurons * 2, dropout_ratio=dropout_ratio, batch_norm=batch_norm)
        # 50 -> 101
        self.uconv1 = up(neurons * 2, neurons * 1, dropout_ratio=dropout_ratio, batch_norm=batch_norm)

        self.outc = outconv(neurons, n_classes)

         # Move network parameters to the specified device
        self.to(device)

        # output info
        print("UNet plastic model with plastic rule [%s] initialized" % self.rule)

    def forward(self, x, hebb):
        # 101 -> 50
        xc1 = self.conv1(x)
        x1 = self.pool1(xc1)
        #print("X1", x1.shape)

        # 50 -> 25
        xc2 = self.conv2(x1)
        x2 = self.pool2(xc2)
        #print("X2", x2.shape)

        # 25 -> 12
        xc3 = self.conv3(x2)
        x3 = self.pool3(xc3)
        #print("X3", x3.shape)

        # 12 -> 6
        xc4 = self.conv4(x3)
        x4 = self.pool4(xc4)
        #print("X4", x4.shape)

        # Middle
        x5 = self.mid(x4)
        #print("X5", x5.shape)

        # 6 -> 12
        x = self.uconv4(x5, xc4)
        #print("X6", x.shape)

        # 12 -> 25
        x = self.uconv3(x, xc3)
        #print("X7", x.shape)

        # 25 -> 50
        x = self.uconv2(x, xc2)
        #print("X8", x.shape)

        # 50 -> 101
        x = self.uconv1(x, xc1)
        #print("X9", x.shape)

        x = self.outc(x)
        #print("OUT", x.shape)

        # The Plasticity rule implementation
        activin = x.view(self.nbf, self.nbf) # The batch size assumed to be 1

        if self.alfa_type == 'free':
            activ = activin.mm(self.w + torch.mul(self.alpha, hebb))
        elif self.alfa_type == 'yoked':
            activ = activin.mm(self.w + self.alpha * hebb)
        else:
            raise ValueError("Must select one plasticity coefficient type ('free' or 'yoked')")

        activout = torch.sigmoid(activ)

        if self.rule == 'hebb':
            hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(activin.unsqueeze(2), activout.unsqueeze(1))[0] # bmm used to implement outer product; remember activs have a leading singleton dimension
        elif self.rule == 'oja':
            hebb = hebb + self.eta * torch.mul((activin[0].unsqueeze(1) - torch.mul(hebb, activout[0].unsqueeze(0))), activout[0].unsqueeze(0)) # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
        else:
            raise ValueError("Must select one learning rule ('hebb' or 'oja')")

        return activout, hebb

    def initialZeroHebb(self):
        """
        Creates variable to store Hebbian plastisity coefficients
        """
        return torch.zeros(self.nbf, self.nbf, dtype=torch.float, device=self.torch_dev)

class conv_module(nn.Module):
    """
    The simple convolution module with optional batch normalization and activation
    """
    def __init__(self, out_ch, kernel_size, stride=1, padding=1, activation=True, batch_norm=False):
        super(conv_module, self).__init__()
        if batch_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

        self.activation = activation
        if activation == True:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activation == True:
            x = self.activ(x)
        return x

class residual_block(nn.Module):
    """
    The residual block
    """
    def __init__(self, out_ch, batch_norm=False):
        super(residual_block, self).__init__()
        if batch_norm == True:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
                conv_module(out_ch, kernel_size=3),
                conv_module(out_ch, kernel_size=3, activation=False)
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                conv_module(out_ch, kernel_size=3),
                conv_module(out_ch, kernel_size=3, activation=False)
            )

    def forward(self, input):
        x = self.conv(input)
        x = x.add(input)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    """
    The ascending convolution module increasing features by 2 in each dimension and
    decreasing channels number by 2 at the same time
    """
    def __init__(self, in_ch, out_ch, dropout_ratio, batch_norm=False):
        super(up, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=0)
        self.uconv =  nn.Sequential(
            nn.Dropout2d(p=dropout_ratio, inplace=True),
            middle(in_ch, out_ch, batch_norm=False)
        )

    def forward(self, x1, x2):
        x = self.dconv(x1)
        diffX = x2.size()[2] - x.size()[2] #TODO Check the correct order
        diffY = x2.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x, x2], dim=1)
        x = self.uconv(x)
        return x


class middle(nn.Module):
    """
    The middle convolution
    """
    def __init__(self, in_ch, out_ch, batch_norm=False):
        super(middle, self).__init__()
        self.mconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            residual_block(out_ch=out_ch, batch_norm=batch_norm),
            residual_block(out_ch=out_ch, batch_norm=batch_norm),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mconv(x)
        return x

class pool_drop(nn.Module):
    """
    The pooling with subsequential dropout
    """
    def __init__(self, dropout_ratio):
        super(pool_drop, self).__init__()
        self.dpool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_ratio, inplace=True)
        )

    def forward(self, x):
        x = self.dpool(x)
        return x


class down(nn.Module):
    """
    The downscending convolution increasing
    channels number by 2
    """
    def __init__(self, in_ch, out_ch, batch_norm=False):
        super(down, self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            residual_block(out_ch=out_ch, batch_norm=batch_norm),
            residual_block(out_ch=out_ch, batch_norm=batch_norm),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.dconv(x)
        return x

#########################################

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

def load_test_dataset(data_dir,
                      img_width,
                      img_height,
                      img_chan,
                      partial=False,
                      part_size=100,
                      debug=False):
    """
    Loads test data set from provided data directory and perform preprocessing
    Arguments:
        data_dir:   The data directory to load test data
        img_width:  The target image width
        img_height: The target image height
        img_chan:   The target image channels
        partial:    The flag to indicate whether only part of dataset should be loaded (usefull for debug)
        part_size:  The size of partial dataset to be loaded if specified
    Returns:
        The preprocessed dataset with test data samples along with names
    """
    test_ids = [name[:-4] for name in next(os.walk(data_dir + "/test/images"))[2]]
    if partial == True:
        test_ids = test_ids[:part_size]

    test_df = pd.DataFrame(index=test_ids)
    test_df["images"] = [np.array(load_image('{}/test/images/{}.png'.format(data_dir, idx), (img_height, img_width))) for idx in test_df.index]

    return test_df

#########################################

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
        start = datetime.fromtimestamp(time.time()).strftime("%B %d, %Y %H:%M:%S")
        stop = datetime.fromtimestamp(params['stop_time']).strftime("%B %d, %Y %H:%M:%S")
        print("Training started at: [%s] and set to stop at: [%s]" %
                (start, stop))

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

        # check if need to force stop training due to time limits or due to epochs limits
        time_is_out = (params['stop_time'] > 0 and next_epoch_finish_time >= params['stop_time'])
        terminate_training = time_is_out or (epoch + 1) == params['epochs']

        if params['debug']:
            print('Epoch finished! Loss: %f, time spent: %d, terminate due to time limits: %s' %
                    (epoch_loss, epoch_time, terminate_training))

        #
        # Perform validation
        #
        if (epoch + 1) % params['val_every'] == 0 or terminate_training:
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
        if (epoch + 1) % params['save_every'] == 0 or terminate_training:
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
            print("Training finished!")
            print("Current epoch %d, train loss: %s" % (epoch, epoch_loss))
            if time_is_out:
                print("Terminated due to the time limit!")
                print("Stop time limit: %d, estimated time of next epoch end: %d" %
                        (params['stop_time'], next_epoch_finish_time))
            break

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
    net = UNetpRes(n_channels=params['im_chan'],
                    neurons=start_neurons,
                    n_classes=1,
                    nbf=img_width,
                    batch_norm=False,
                    bilinear_upsample=False,
                    device=device,
                    rule=prule)

    if load:
        net.load_state_dict(torch.load(model))
        net.to(device)
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

#########################################

def encode(im):
    """
    Performs RLE encoding of provided binary mask image data with shape (r,c)
    Arguments:
        img: is binary mask image, shape (r,c)
    Returns: run length as an array or string (if format is True)
    """
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#########################################

def inference(net,
              img_data,
              device):
    """
    Do mask inference for provided image.
    Arguments:
        net:            The trained network
        img_data:       The input image data for inference
        device:         The Torch device to be used
        mask_threshold: The minimum probability value to consider a mask pixel white
    Return:
        the predicted mask image
    """
    net.eval()
    with torch.no_grad():
        hebb = net.initialZeroHebb()

        t_img = torch.from_numpy(np.array([img_data.astype(np.float32)])).to(device)
        y_pred, _ = net(Variable(t_img, requires_grad=False), Variable(hebb, requires_grad=False))

    mask = y_pred.squeeze().cpu().numpy()
    return mask

def predict(net,
            test_df,
            params,
            visualize=False,
            save_masks=False):
    """
    Iterate over all test images and do masks prediction
    Arguments:
        net:            The trained network model for inference
        test_df:        The test data samples
        params:         The dictionary with parameters
        visualize:      The flag to indicate whether to visualize the images as they are processed
        save_masks:     The flag to indicate whether to save the output masks
    """
    print("Start prediction with the number of test image samples:", len(test_df.index))
    print(params)

    # Extracting test samples
    X_test = np.array(test_df.images.tolist()).reshape(-1, params['img_chan'], params['img_height'], params['img_width'])
    mask_threshold = params['mask_threshold']

    # iterate over test images and do inference
    preds_downsampled = []
    for i, img_data in enumerate(X_test):
        mask = inference(net=net,
                         img_data=np.array(img_data),
                         device=net.torch_dev)
        preds_downsampled.append(mask)

        if visualize:
            image = img_data.squeeze()
            mask_t = (mask > mask_threshold).astype(np.uint8)
            plot_image_mask(np.dstack((image,image,image)), mask_t)

        if save_masks:
            if not os.path.isdir(params['out_dir']):
                os.mkdir(params['out_dir'])

            mask_t = (mask > mask_threshold).astype(np.uint8)
            tmp = np.squeeze(mask_t).astype(np.float32)
            out_path = "%s/masks/%s.png" % (params['out_dir'], test_df.index[i])
            imsave(out_path, np.dstack((tmp,tmp,tmp)))

    # Sanity check
    print(preds_downsampled[0].shape)

    print("Inference complete")

    # prepare submission data in CSV file as RLE encoded
    pred_dict = {fn:encode(np.round(preds_downsampled[i] > mask_threshold)) for i,fn in enumerate(test_df.index)}

    # save to submission file
    subm_file = params['out_dir'] + "/" + params['subm_file']
    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(subm_file)

    print("Results encoded to:", subm_file)

def start_inference(model,
                    test_df,
                    X_valid,
                    y_valid,
                    out_dir,
                    img_width,
                    img_height,
                    img_chan,
                    subm_file="submission.csv",
                    gpu=True,
                    visualize=False,
                    save_masks=False,
                    debug=False):
    """
    Starts inference by loading trained network model
    Arguments:
        model:          The trained network model and state dictionary
        test_df:        The test data samples along with names
        X_valid:        The data samples for validation (used for best thershold evaluation)
        y_valid:        The ground truth data for validation (used for best thershold evaluation)
        out_dir:        The directory to save results
        subm_file:      The file name of submission file
        gpu:            The flag to indicate whether to use GPU for inference
        img_width:      The width of the resized image
        img_height:     The height of the resized image
        img_chan:       The number of channels in input plot_image
        visualize:      The flag to indicate whether to visualize the images as they are processed
        save_masks:     The flag to indicate whether to save the output masks
        debug:          The flag to indicate whether to show debug information
    """
    # Create torch device for tensor operations
    device = None
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    net = UNetpRes(n_channels=img_chan,
                    neurons=start_neurons,
                    n_classes=1,
                    nbf=img_width,
                    device=device)

    print("Loading model %s" % (model))
    net.load_state_dict(torch.load(model))
    net.to(device)

    # Best threshold evaluation
    print("Score model for best IoU")
    threshold_best, iou_best = score_model_best_iou(net=net,
                                                    X_valid=X_valid,
                                                    y_valid=y_valid,
                                                    device=device,
                                                    debug=debug)
    print("Best threshold: %f, best IoU: %f" % (threshold_best, iou_best))

    # Put parameters into dictionary
    params = {"out_dir":out_dir,
              "device":device,
              "img_width":img_width,
              "img_height":img_height,
              "img_chan":img_chan,
              "mask_threshold":threshold_best,
              "subm_file":subm_file,
              "debug":debug}

    predict(net=net,
            test_df=test_df,
            params=params,
            visualize=visualize,
            save_masks=save_masks)

#########################################

def score_model_best_iou(net, X_valid, y_valid, device, debug=False):
    """
    Scores the model and do a threshold optimization by the best IoU.
    Arguments:
        net:        The network to be evaluated
        X_valid:    The data samples for validation
        y_valid:    The ground truth data for validation
        device:     The Torch device to use
        debug:      The flag to indicate if debug info should be displayed
    Returns:
        (threshold_best, iou_best) the threshold for best IoU metric and best IoU metric value
    """
    net.eval()

    with torch.no_grad():
        hebb = net.initialZeroHebb()

        # Find predictions for validation
        preds_valid = []
        for x in X_valid:
            t_img = torch.from_numpy(np.array([x.astype(np.float32)])).to(device)

            # We do not learn plasticity within validation
            mask_pred, _ = net(Variable(t_img, requires_grad=False), Variable(hebb, requires_grad=False))

            preds_valid.append(mask_pred.cpu().numpy())

        # Scoring model, choose threshold by validation data
        thresholds_ori = np.linspace(0.3, 0.7, 31)
        # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
        thresholds = np.log(thresholds_ori/(1-thresholds_ori))

        ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in thresholds])
        if debug:
            print(ious)

        # instead of using default 0 as threshold, use validation data to find the best threshold.
        threshold_best_index = np.argmax(ious)
        iou_best = ious[threshold_best_index]
        threshold_best = thresholds[threshold_best_index]

        if debug:
            plot_best_iou(thresholds=thresholds, ious=ious)

        return threshold_best, iou_best

#########################################
# Do training
#########################################
# Set input parameters
input_data_dir="../input"
data_dir=input_data_dir + "/tgs-salt-identification-challenge"
output_dir="."
model=""
load_model=False
use_gpu=True
training_epochs=250#2000000
learning_rate=3e-4#3e-5
step_lr=1e4#1e5
save_every=20
validate_every=1
rollout_every=100
show_debug=False#True
t_img_width=101
t_img_height=101
t_img_chan=1

plastic_rule='hebb'#'oja'#

max_train_time=5*3600#3*3600#14500#21000#19600 #

do_train = True#False#
do_inference = False#True#

short_run = False#True#
short_size = 100

# Load training dataset
x_train, x_valid, y_train, y_valid = load_train_dataset(data_dir=data_dir,
                                                        img_width=t_img_width,
                                                        img_height=t_img_height,
                                                        img_chan=t_img_chan,
                                                        debug=show_debug)

if do_train:
    print("Starting training with rule: %s, start neurons: %d" % (plastic_rule, start_neurons))

    # First test on small dataset
    if short_run:
        x_train = x_train[:short_size,:,:,:]
        x_valid = x_valid[:short_size,:,:,:]
        y_train = y_train[:short_size,:,:,:]
        y_valid = y_valid[:short_size,:,:,:]

    # start network training
    start_train(x_train, x_valid, y_train, y_valid,
                out_dir=output_dir,
                model=model,
                load=load_model,
                gpu=use_gpu,
                epochs=training_epochs,
                lr=learning_rate,
                steplr=step_lr,
                max_train_time=max_train_time,
                save_every=save_every,
                val_every=validate_every,
                rollout=rollout_every,
                prule=plastic_rule,
                img_width=t_img_width,
                img_height=t_img_height,
                img_chan=t_img_chan,
                debug=show_debug)

    print("Training complete")

###################################
# Do prediction
###################################
model_file=input_data_dir + "/tgs-unet-plastic-res-hebb/train_net_res.pth"
subm_file="submission_res-1.csv"

if do_inference:
    print("Starting inference with model:%s, start neurons: %d" % (model_file, start_neurons))

    print('Getting and resizing test images... ')
    test_df = load_test_dataset(data_dir=data_dir,
                               img_width=t_img_width,
                               img_height=t_img_height,
                               img_chan=t_img_chan,
                               partial=short_run,
                               part_size=short_size,
                               debug=show_debug)
    print("Done!")

    start_inference(model=model_file,
                    test_df=test_df,
                    X_valid=x_valid,
                    y_valid=y_valid,
                    out_dir=output_dir,
                    subm_file=subm_file,
                    gpu=use_gpu,
                    img_width=t_img_width,
                    img_height=t_img_height,
                    img_chan=t_img_chan,
                    visualize=False,
                    save_masks=False,
                    debug=show_debug)

    print("Prediction complete")
