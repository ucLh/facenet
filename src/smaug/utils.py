from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import scipy.misc
import tensorflow as tf


def load_checkpoint(ckpt_dir_or_file, session, restorer):
    print('Loading checkpoint...')
    if os.path.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    restorer.restore(session, ckpt_dir_or_file)
    print('Loading succeeds! Copy variables from % s' % ckpt_dir_or_file)


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def _to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    # transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def imwrite(image, path):
    # save an [-1.0, 1.0] image
    return scipy.misc.imsave(path, _to_range(image, 0, 255, np.uint8))


def immerge(images, row, col):
    """Merge images.

    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    if images.ndim == 4:
        c = images.shape[3]
    elif images.ndim == 3:
        c = 1

    h, w = images.shape[1], images.shape[2]
    if c > 1:
        img = np.zeros((h * row, w * col, c))
    else:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img
