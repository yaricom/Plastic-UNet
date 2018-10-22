#!/bin/sh
#
# The script to run Keras history visualization
#

help () {
    echo
    echo "The script to visualize collected Keras history"
    echo "Usage:"
    echo "     keras_history_plot.sh data_file"
    echo "          data_file - the pickle dump with Keras history"
    echo
}

if [[ "$#" -lt 1 ]]; then
    help
    exit 0
fi

/usr/bin/env python3 src/utils/keras_history_visualization.py --data-file $1
