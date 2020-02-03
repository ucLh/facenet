"""Uses trained model to calculate embeddings and saves them in a feature_vectors.txt file"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
import os.path as p
import argparse
import facenet
import cv2


def main(args):
    # Load data
    dataset = facenet.get_dataset(args.data_root)
    image_paths, _ = facenet.get_image_paths_and_labels(dataset)
    nrof_images = len(image_paths)

    for i in range(nrof_images):
        path = os.path.abspath(image_paths[i])
        print(path)
        class_name = get_class_name(path)
        image_name = get_image_name(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (286, 286))
        save_path = '/home/luch/Programming/Python/facenet/datasets/resized/' + class_name + '/' + image_name
        dir_name = '/home/luch/Programming/Python/facenet/datasets/resized/' + class_name

        if not os.path.isdir(dir_name):  # Create the log directory if it doesn't exist
            os.makedirs(dir_name)

        cv2.imwrite(save_path, img)
        # print(save_path)


def get_image_name(path):
    return p.split(path)[-1]


def get_class_name(path):
    return p.split(p.dirname(path))[-1]


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str,
                        help='Path to data directory which needs to forward passed through the network',
                        default='../datasets/current_train/')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf '
                             '(.pb) file',
                        default='../models/test_pb/optimized_val_84.pb')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.',
                        default=256)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
