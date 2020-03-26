"""Uses feature_vectors.txt file produced by process_database.py to match an input image with an image in the
   database"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
import argparse
import facenet
import json
import numpy as np
from PIL import Image
from scipy import misc


def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch_p:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load and preprocess image
            original_img = misc.imread(args.image_path, mode='RGB')
            img = misc.imresize(original_img, (args.image_size, args.image_size), interp='bilinear')
            img = np.expand_dims(img, axis=0)
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 2 - 1

            # Calculate target image embedding
            target_emb = sess.run(embeddings, feed_dict={images_placeholder: img, phase_train_placeholder: False})

            min_dist = None
            min_path = None

            with open("feature_vectors.txt", "r") as file:

                for line in file:

                    # Calculate embedding for each image in the database
                    line = line.strip()
                    emb_dict = json.loads(line)
                    path, emb = list(emb_dict.items())[0]
                    emb = np.array(emb)

                    # Then calculate distance to the target
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb, target_emb[0]))))
                    if (min_dist is None) or (dist < min_dist):
                        min_dist = dist
                        min_path = path

            print(min_dist, min_path)

            closest_img = misc.imread(min_path, mode='RGB')
            show_result(original_img, closest_img)


def show_result(original_img, closest_img):
    merged_img = resize_and_merge(original_img, closest_img)
    output = Image.fromarray(merged_img)
    output.show()


def resize_and_merge(img_a, img_b):
    """Merges 2 images. Images should be passed as arrays"""
    shape_a, shape_b = img_a.shape, img_b.shape
    new_width = min(shape_a[0], shape_b[0], 1024)
    new_height = min(shape_a[1], shape_b[1], 512)

    resized_img_a = misc.imresize(img_a, (new_width, new_height), interp='bilinear')
    resized_img_b = misc.imresize(img_b, (new_width, new_height), interp='bilinear')

    assert resized_img_a.shape == resized_img_b.shape

    merged_img = np.concatenate((resized_img_a, resized_img_b), axis=1)

    return merged_img


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str, help='Image to compare')
    parser.add_argument('--data_root', type=str,
                        help='Path to data directory which needs to be forward passed through the network',
                        default='../datasets/series_for_test/')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf '
                             '(.pb) file',
                        default='../models/val84-wout-queries')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.',
                        default=256)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
