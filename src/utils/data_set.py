# The script to load data set and do split it accordingly
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils import load_image

from utils import plot_coverage
from utils import plot_depth

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
