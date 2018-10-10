# The script to visualize data collected during training session
from optparse import OptionParser

import h5py

import numpy as np
import matplotlib.pyplot as plt

def render_data(hdf5_file, window_size = 1000):
    """
    Renders data collected during training session
    Arguments:
        hdf5_file:      The file with data points collected
        window_size:    The averaging window size
    """
    # read input data
    with h5py.File(hdf5_file, 'r') as f:
        val_train_losses = f["validation/train_losses"][()]
        val_test_losses = f["validation/test_losses"][()]
        val_accuracies = f["validation/accuracies"][()]

        all_losses = f["train/all_losses"][()]

    n_epochs = all_losses.shape[0]
    n_val_points = val_train_losses.shape[0]
    print("Number of epochs: %d, number of validation points: %d" % (n_epochs, n_val_points))

    fig = plt.figure()
    # Render validation data points
    if n_val_points > 0:
        plt.subplot(2, 1, 1)
        plt.xlim([0, n_val_points])
        x = np.arange(n_val_points)
        plot_with_average(x, val_train_losses, style='r', label='Train Loss', window=window_size)
        plot_with_average(x, val_test_losses, style='b', label='Validation Loss', window=window_size)
        plot_with_average(x, val_accuracies, style='g', label='Validation accuracy', window=window_size)
        plt.ylabel('loss')
        plt.legend()
        plt.title("Validation data")

    if n_epochs > 0:
        plt.subplot(2, 1, 2)
        plt.xlim([0, n_epochs])
        x = np.arange(n_epochs)
        plot_with_average(x, all_losses, style='r', label='Train Loss', window=window_size)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title("Training loss")

    plt.show()

def plot_with_average(x, y, style, label, window):
    plt.plot(x, y, style, alpha=0.5)
    y_av = moving_average(y, window)
    plt.plot(x, y_av, style, label=label)


def moving_average(data, window_size):
    """
    Returns moving average for specified data given specific window size.
    Arguments:
        data:           The data points
        window_size:    The averaging window size
    """
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def parse_args():
    """
    Parses command line arguments
    """
    parser = OptionParser()
    parser.add_option('--data-file', '-f', default='train_data.hdf5',
                        help="The path to the data file")
    parser.add_option('--avg-window-size', '-w', default='1', type='int',
                        help="The window size for moving average")

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = parse_args()

    render_data(args.data_file, args.avg_window_size)
