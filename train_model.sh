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

/usr/bin/env python3 src/train.py --epochs 10 --save_every 5 --validate_every 5 \
                                  --learning-rate 3e-5 --step-lr 1e6 \
                                  --max-train-time 2000 --rollout_every 5000\
                                  --prule oja \
                                  --data ./data1 --out ./out \
#                                  --dataset ./data/dataset.hdf5 --out ./out
