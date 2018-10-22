# The script to visualize Keras history data
from optparse import OptionParser
import pickle

import matplotlib.pyplot as plt
import numpy as np

def plot_history(hist_file):
    """
    Plots historic data from Keras learning routine stored as pickde dump
    Arguments:
        hist_file: The pickle dump with historic data for analysis
    """
    # Load historic data from pickle
    with open(hist_file, 'rb') as fo:
        history = pickle.load(fo)

    # list all data in history
    print(history.keys())

    nsubplots=2
    fig, axes = plt.subplots(ncols=nsubplots)

    # summarize history for accuracy
    axes[0].plot(history['mean_iou'])
    axes[0].plot(history['val_mean_iou'])
    axes[0].set_title('model accuracy')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    axes[1].plot(history['loss'])
    axes[1].plot(history['val_loss'])
    axes[1].set_title('model loss')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.show()

def parse_args():
    """
    Parses command line arguments
    """
    parser = OptionParser()
    parser.add_option('--data-file', '-f',
                        help="The path to the data file")

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = parse_args()

    plot_history(hist_file=args.data_file)
