# The network evaluation

import torch
from torch.autograd import Variable
from torch.autograd import no_grad

import numpy as np

from utils import iou_metric


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
            acc = iou_metric(y_pred_in=y_pred_flat.cpu().numpy(), y_true_in=y_target_flat.cpu().numpy(), print_table=debug)
            total_acc += acc

    return (total_acc / (i + 1), val_loss / (i + 1))
