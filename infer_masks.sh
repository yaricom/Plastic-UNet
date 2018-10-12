#!/bin/sh
#
# The mask images inference runner
#
help () {
    echo
    echo "The masks inference script"
    echo "Usage:"
    echo "      infer_masks.sh"
    echo
}

/usr/bin/env python3 src/infer.py --model ./out/train_net.pth \
                                  --data ./data2 --out ./out \
                                  --visualize True --save False
