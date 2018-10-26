# The network evaluation
from optparse import OptionParser

import torch
from torch.autograd import Variable
from torch.autograd import no_grad

import numpy as np
import matplotlib.pyplot as plt

from utils import iou_metric
from utils import iou_metric_batch
from utils import fast_iou_metric
from utils import plot_best_iou
from utils import load_train_dataset

from unet import UNetpRes


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


def get_args():
    """
    Parses command line arguments
    """
    parser = OptionParser()
    parser.add_option('--model', '-m', default='MODEL.pth',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_option('-i', '--data', dest='data_dir', type='string',
                      help='the directory with input test data')

    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')

    parser.add_option('-v', '--debug', action='store_true', dest='debug',
                      default=False, help='show debug information')

    (options, args) = parser.parse_args()
    return options

if __name__ == "__main__":
    args = get_args()

    # Set values
    img_width=101
    img_height=101
    img_chan=1

    # Create torch device for tensor operations
    device = None
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    net = UNetpRes(n_channels=img_chan,
                    n_classes=1,
                    device=device,
                    nbf=img_width)

    print("Loading model %s" % (args.model))
    net.load_state_dict(torch.load(args.model))
    net.to(device)

    print("Loading data set")
    x_train, x_valid, y_train, y_valid = load_train_dataset(data_dir=args.data_dir,
                                                            img_width=img_width,
                                                            img_height=img_height,
                                                            img_chan=img_chan,
                                                            debug=args.debug)
    # Do best thershold caclulation
    print("Calculating best thershold value")
    threshold_best, iou_best = score_model_best_iou(net,
                                                    x_valid,
                                                    y_valid,
                                                    device,
                                                    debug=args.debug)
    print("Best thershold:", threshold_best)
    print("Best IoU", iou_best)
