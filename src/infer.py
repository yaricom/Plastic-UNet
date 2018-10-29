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
from skimage.io import imsave

from unet import UNetp
from unet import UNetpRes

from utils import hwc_to_chw
from utils import plot_image_mask
from utils import plot_test_check
from utils import load_test_dataset

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
                    out_dir,
                    img_width,
                    img_height,
                    img_chan,
                    mask_threshold,
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
        out_dir:        The directory to save results
        subm_file:      The file name of submission file
        gpu:            The flag to indicate whether to use GPU for inference
        img_width:      The width of the resized image
        img_height:     The height of the resized image
        img_chan:       The number of channels in input plot_image
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

    net = UNetpRes(n_channels=img_chan,
                    n_classes=1,
                    nbf=img_width,
                    device=device)

    print("Loading model %s" % (model))
    net.load_state_dict(torch.load(model))
    net.to(device)

    # Put parameters into dictionary
    params = {"out_dir":out_dir,
              "device":device,
              "img_width":img_width,
              "img_height":img_height,
              "img_chan":img_chan,
              "mask_threshold":mask_threshold,
              "subm_file":subm_file,
              "debug":debug}

    predict(net=net,
            test_df=test_df,
            params=params,
            visualize=visualize,
            save_masks=save_masks)

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
    parser.add_option('--save', '-s', action='store_true',
                        help="To save the output masks",
                        default=False)
    parser.add_option('--mask-threshold', '-t', dest='mask_threshold', type=float,
                        help="Minimum probability value to consider a mask pixel white")

    parser.add_option('--partial', '-p', action='store_true',
                        help="To run on partial dataset", default=False)
    parser.add_option('--partial-size', '-d', dest='partial_size', default=100, type='int',
                      help='The size of partial dataset')

    (options, args) = parser.parse_args()
    return options

if __name__ == "__main__":
    args = get_args()

    # Check if output directory exists
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # Set values
    t_img_width=101
    t_img_height=101
    t_img_chan=1

    # Load test dataset
    if args.data_dir != None:
        # Get test images
        print('Getting and resizing test images... ')
        test_df = load_test_dataset(data_dir=args.data_dir,
                                   img_width=t_img_width,
                                   img_height=t_img_height,
                                   img_chan=t_img_chan,
                                   partial=args.partial,
                                   part_size=args.partial_size,
                                   debug=False)

        print('Done!')
    else:
        raise ValueError("The input data directory or dataset file not specified")

    start_inference(model=args.model,
                    test_df=test_df,
                    out_dir=args.out_dir,
                    gpu=args.gpu,
                    img_width=t_img_width,
                    img_height=t_img_height,
                    img_chan=t_img_chan,
                    mask_threshold=args.mask_threshold,
                    visualize=args.visualize,
                    save_masks=args.save,
                    debug=True)
