#!/bin/sh
#
# The script to visualize training data points collected
#
help () {
    echo
    echo "The script to visualize collected train data points from HDF5"
    echo "Usage:"
    echo "      vis_train_data.sh data_file"
    echo "          data_file - the HDF5 file with train data points"
    echo
}

if [[ "$#" -lt 1 ]]; then
    help
    exit 0
fi

w_size=1000
if [[ "$#" -ge 2 ]]; then
    w_size=$2
fi

runs=-1
if [[ "$#" -eq 3 ]]; then
    runs=$3
fi

/usr/bin/env python3 src/utils/data_visualization.py --data-file $1 --avg-window-size $w_size --runs-per-epoch $runs
