# The script to perform inference using trained model
import sys
import os
from optparse import OptionParser

import torch
from torch.autograd import Variable
from torch.autograd import no_grad

import h5py
import numpy as np
from skimage.transform import resize
from skimage.io import imsave

from unet import UNetp
from utils import hwc_to_chw
from utils import chw_to_hwc
from utils import plot_image_mask

# Set some parameters
out_im_width = 101
out_im_height = 101
im_chan = 3
debug = True

def inference(net,
              img_data,
              device,
              mask_threshold=0.5):
    """
    Do mask inference for provided image.
    Arguments:
        net: The trined model
        img_data: The input image data for inference
        device: The Torch device to be used
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
    return mask > mask_threshold

def main(args):
    """
    Iterate over all test images and do masks inference
    """
    # Check if output directory exists
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # Create torch device for tensor operations
    args.device = None
    if args.gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    net = UNetp(n_channels=im_chan, n_classes=1, device=args.device)
    print("Loading model %s" % (args.model))

    net.load_state_dict(torch.load(args.model))
    net.to(args.device)

    # Get test images
    print('Getting test imagesfrom dataset... ')
    sys.stdout.flush()
    with h5py.File(args.data_file, 'r') as f:
        X = f['test/images'][()]
    print('Done!')

    # transpose HWC to CHW image data format accepted by Torch
    X_test = list(map(hwc_to_chw, X))

    # iterate over test images and do inference
    preds_downsampled = []
    for i, img_data in enumerate(X_test):
        mask = inference(net=net,
                         img_data=img_data,
                         device=args.device,
                         mask_threshold=args.mask_threshold)

        mask = resize(mask, (out_im_height, out_im_width), mode='constant')
        preds_downsampled.append(mask)

        if args.visualize:
            image = resize(img_data, (im_chan, out_im_height, out_im_width), mode='constant')
            image = np.transpose(image, (1, 2, 0))
            plot_image_mask(image, mask)

        if args.save:
            out_path = args.out_dir + "/masks/"
            if not os.path.isdir(out_path):
                os.mkdir(out_path)

            imsave(out_path + str(i) + ".png", mask)



def get_args():
    """
    Parses command line arguments
    """
    parser = OptionParser()
    parser.add_option('--model', '-m', default='MODEL.pth',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_option('-i', '--data', dest='data_file', type='string',
                      help='the path to the dataset file with input data')

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
    main(args)
