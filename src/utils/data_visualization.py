# The script to visualize data collected during training session
from optparse import OptionParser

import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

def plot_best_iou(thresholds, ious):
    """
    Plots thersholds vs IoU and best thershold
    """
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()

    plt.show()

def plot_coverage(train_df):
    """
    Plots salt coverage
    Arguments:
        train_df: The data frame with train data coverage inputs
    """
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    sns.distplot(train_df.coverage, kde=False, ax=axs[0])
    sns.distplot(train_df.coverage_class, bins=10, kde=False, ax=axs[1])
    plt.suptitle("Salt coverage")
    axs[0].set_xlabel("Coverage")
    axs[1].set_xlabel("Coverage class")

    plt.show()

def plot_depth(train_df, test_df):
    """
    Plots depth distribution
    Arguments:
        train_df:   The data frame with train data inputs
        test_df:    The data frame with test data inputs
    """
    sns.distplot(train_df.z, label="Train")
    sns.distplot(test_df.z, label="Test")
    plt.legend()
    plt.title("Depth distribution")

    plt.show()

def render_data(hdf5_file, runs_per_epoch, window_size=1000):
    """
    Renders data collected during training session
    Arguments:
        hdf5_file:      The file with data points collected
        runs_per_epoch: The number of runs per epoch (train samples count)
        window_size:    The averaging window size
    """
    # read input data
    with h5py.File(hdf5_file, 'r') as f:
        val_train_losses = f["validation/train_losses"][()]
        val_test_losses = f["validation/test_losses"][()]
        val_accuracies = f["validation/accuracies"][()]

        all_losses = f["train/all_losses"][()]

    n_runs = all_losses.shape[0]
    n_val_points = val_train_losses.shape[0]
    print("Total number of runs: %d, number of validation points: %d, runs per epoch: %d, window: %d"
            % (n_runs, n_val_points, runs_per_epoch, window_size))

    nsubplots = 0
    if n_val_points > 0:
        nsubplots = 1
    if n_runs > 0:
        nsubplots += 1


    fig, axes = plt.subplots(ncols=nsubplots)

    # Render validation data points
    if n_val_points > 0:
        # Convert validation points to pandas data frame
        df = pd.DataFrame(val_train_losses, columns=['Train Loss'])
        df['Test Loss'] = pd.Series(val_test_losses, index=df.index)
        df['Accuracy'] = pd.Series(val_accuracies, index=df.index)

        axes[0].set_xlim([0, n_val_points])
        ax = df.plot(secondary_y=['Accuracy'], ax=axes[0], style=['b', 'g', 'r'])
        axes[0].set_ylabel('loss')
        axes[0].set_xlabel('epochs')
        axes[0].right_ax.set_ylabel('accuracy')
        axes[0].set_title("Validation Data")
    else:
        axes = [None, axes]

    if n_runs > 0:
        if runs_per_epoch > 0:
            axes[1].set_xlim(runs_per_epoch, n_runs)
        else:
            axes[1].set_xlim(0, n_runs)

        x = np.arange(n_runs)

        plot_with_average(x, all_losses, ax=axes[1], style='r', label='Train Loss', window=window_size)
        axes[1].set_xlabel('runs')
        axes[1].set_ylabel('loss')
        axes[1].set_title("Training Costs")

    plt.tight_layout()
    plt.show()

def plot_with_average(x, y, ax, style, label, window):
    ax.plot(x, y, style, alpha=0.5)
    y_av = moving_average(y, window)
    ax.plot(x, y_av, style, label=label)


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
    parser.add_option('--runs-per-epoch', '-r', type='int',
                        help="The number of runs per epoch (train samples count)")
    parser.add_option('--avg-window-size', '-w', default='1', type='int',
                        help="The window size for moving average")

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = parse_args()

    render_data(hdf5_file=args.data_file,
                runs_per_epoch=args.runs_per_epoch,
                window_size=args.avg_window_size)
