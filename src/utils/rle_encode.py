# The script to encode bitmap masks using run-length encoding (RLE).
# https://en.wikipedia.org/wiki/Run-length_encoding

import numpy as np

def encode(img, order='F', format=True):
    """
    Performs RLE encoding of provided binary mask image data with shape (r,c)
    Arguments:
        img: is binary mask image, shape (r,c)
        order: is down-then-right, i.e. Fortran format determines if the order
               needs to be preformatted (according to submission rules) or not
        format: If True than run lenght will be returned as formatted string, otherwise
                array with values will be retruned

    Returns: run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
