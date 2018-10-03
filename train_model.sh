#!/bin/sh
#
# The model training runner
#

help () {
    echo
    echo "The model training script"
    echo "Usage:"
    echo "      train_model.sh"
    echo
}

/usr/bin/env python3 src/train.py --epochs 5 --save_every 5 --data ./data/train --out ./out