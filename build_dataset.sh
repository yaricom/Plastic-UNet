#!/bin/sh
#
# The model training runner
#

help () {
    echo
    echo "The script to build resized images dataset as HDF5"
    echo "Usage:"
    echo "      build_dataset.sh data_dir"
    echo "          data_dir - the directory to look for training and test data files"
    echo
}

if [[ "$#" -lt 1 ]]; then
    help
    exit 0
fi

/usr/bin/env python3 src/utils/img_utils.py --data $1
