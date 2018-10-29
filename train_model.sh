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

/usr/bin/env python3 src/train.py --epochs 5 --save_every 50 --validate_every 1 \
                                  --learning-rate 3e-4 --step-lr 1e5 \
                                  --max-train-time -1 --rollout_every 100\
                                  --prule hebb \
                                  --data ./data1 --out ./out --debug
#                                  --dataset ./data/dataset.hdf5 --out ./out
