# The script to perform inference using trained model
import sys
import os
from optparse import OptionParser

import torch
from torch.autograd import Variable
from torch.autograd import no_grad

import h5py
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imsave

from unet import UNetp

from utils import hwc_to_chw
from utils import plot_image_mask
from utils import plot_test_check
from utils import load_image

from utils import encode

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
            data_dir,
            out_dir,
            test_ids,
            subm_file="submission.csv",
            mask_threshold=0.5,
            visualize=False,
            save_masks=False,
            img_width=128,
            img_height = 128,
            out_img_width=101,
            out_img_height=101,
            img_chan=3,
            debug=False):
    """
    Iterate over all test images and do masks prediction
    Arguments:
        net:            The trained network model for inference
        data_dir:       The directory to look for test data
        out_dir:        The directory to save results
        test_ids:       The array with IDs of test images to use
        subm_file:      The file name of submission file
        mask_threshold: The minimum probability value to consider a mask pixel white
        visualize:      The flag to indicate whether to visualize the images as they are processed
        save_masks:     The flag to indicate whether to save the output masks
        img_width:      The width of the resized image
        img_height:     The height of the resized image
        out_img_width:  The width of the original image
        out_img_height: The height of the original image
        img_chan:       The number of channels in input plot_image
        debug:          The flag to indicate whether to show debug information
    """
    # Get test images
    print('Getting and resizing test images... ')
    X_test = np.zeros((len(test_ids), img_height, img_width, img_chan), dtype=np.float64)
    for n, id_ in enumerate(test_ids):
        x = load_image(data_dir + '/test/images/' + id_, (img_height, img_width, img_chan))
        X_test[n] = x

    print('Done!')

    #
    # Check if test data looks all right
    #
    #if debug:
    #    plot_test_check(X_test)

    # transpose HWC to CHW image data format accepted by Torch
    X_test = list(map(hwc_to_chw, X_test))

    # iterate over test images and do inference
    preds_downsampled = []
    for i, img_data in enumerate(X_test):
        mask = inference(net=net,
                         img_data=img_data,
                         device=net.torch_dev)

        mask = resize(mask, (out_img_height, out_img_width), mode='constant', preserve_range=True)
        preds_downsampled.append(mask)

        if visualize:
            image = resize(img_data, (img_chan, out_img_height, out_img_width), mode='constant')
            image = np.transpose(image, (1, 2, 0))
            mask_t = (mask > mask_threshold).astype(np.uint8)
            plot_image_mask(image, mask_t)

        if save_masks:
            out_path = args.out_dir + "/masks/"
            if not os.path.isdir(out_path):
                os.mkdir(out_path)

            mask_t = (mask > mask_threshold).astype(np.uint8)
            tmp = np.squeeze(mask_t).astype(np.float32)
            imsave(out_path + test_ids[i], np.dstack((tmp,tmp,tmp)))

    # Sanity check
    print(preds_downsampled[0].shape)

    # prepare submission data in CSV file as RLE encoded
    pred_dict = {fn[:-4]:encode(np.round(preds_downsampled[i])) for i,fn in enumerate(test_ids)}

    # save to submission file
    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(out_dir + "/" + subm_file)

def start_inference(model,
                    data_dir,
                    out_dir,
                    gpu=True,
                    mask_threshold=0.5,
                    visualize=False,
                    save_masks=False,
                    img_chan=3,
                    debug=False):
    """
    Starts inference by loading trained network model
    Arguments:
        model:          The trained network model and state dictionary
        data_dir:       The directory to look for test data
        out_dir:        The directory to save results
        subm_file:      The file name of submission file
        gpu:            The flag to indicate whether to use GPU for inference
        mask_threshold: The minimum probability value to consider a mask pixel white
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

    net = UNetp(n_channels=img_chan, n_classes=1, device=device)

    print("Loading model %s" % (model))
    net.load_state_dict(torch.load(model))
    net.to(device)

    test_ids = next(os.walk(data_dir + "/test/images"))[2]
    print("Test images count: %d" % len(test_ids))

    predict(net=net,
            data_dir=data_dir,
            out_dir=out_dir,
            test_ids=test_ids,
            mask_threshold=mask_threshold,
            visualize=visualize,
            save_masks=save_masks,
            img_chan=img_chan,
            debug=debug)

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

    parser.add_option('--out', '-o', dest='out_dir', default='./out',
                        help='directory for ouput images')

    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')

    parser.add_option('--visualize', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_option('--save', '-n', action='store_true',
                        help="To save the output masks",
                        default=False)
    parser.add_option('--mask-threshold', '-t', dest='mask_threshold', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    (options, args) = parser.parse_args()
    return options

if __name__ == "__main__":
    args = get_args()

    # Check if output directory exists
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    start_inference(model=args.model,
                    data_dir=args.data_dir,
                    out_dir=args.out_dir,
                    gpu=args.gpu,
                    mask_threshold=args.mask_threshold,
                    visualize=args.visualize,
                    save_masks=args.save,
                    debug=True)
