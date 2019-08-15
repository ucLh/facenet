# a, b, c, d, e = ImageFile('a', 1), ImageFile('b', 2), ImageFile('c', 3), ImageFile('d', 4), ImageFile('e', 5)
# img_file_list = [a, b, c, d, e]
# img_file_list = insert_element(ImageFile('aa', 1), img_file_list, upper_bound=5)
# print(list(map(str, img_file_list)))

"""Uses feature_vectors.txt file produced by process_database.py to match an input image with an image in the
   database"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
from os import path as p
import argparse
import facenet
import json
import numpy as np
from PIL import Image
from scipy import misc
from time import time

from smaug.data import ImageData


def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch_p:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load data
            dataset = facenet.get_dataset(args.data_root)
            image_paths, _ = facenet.get_image_paths_and_labels(dataset)
            images = ImageData(image_paths, sess, load_size=args.image_size, batch_size=1)
            nrof_images = len(images)
            count = 0

            emb_array = None
            path_list = []
            line_count = 0
            with open(args.feature_vectors_file, "r") as file:
                for line in file:
                    # Calculate embedding
                    emb, path = get_embedding_and_path(line)

                    if line_count % 100 == 0:
                        print(path)
                    line_count += 1

                    if emb_array is None:
                        emb_array = np.array(emb)
                    else:
                        emb_array = np.concatenate((emb_array, emb))
                    path_list.append(path)

            emb_array = emb_array.reshape((-1, 512))
            print(emb_array.shape)
            for i in range(nrof_images):
                img = images.batch()
                target_path = p.abspath(image_paths[i])
                target_class_name = get_querie_class_name(target_path)

                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                target_emb = sess.run(embeddings, feed_dict=feed_dict)

                img_file_list = []
                upper_bound = args.top_n
                # start_time = time()

                for j in range(len(emb_array)):

                    # Then calculate distance to the target
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb_array[j], target_emb[0]))))

                    # Insert a score with a path
                    img_file = ImageFile(path_list[j], dist)
                    img_file_list = insert_element(img_file, img_file_list, upper_bound=upper_bound)

                class_name = get_querie_class_name(img_file_list[0].path)
                if class_name == target_class_name:
                    print(target_class_name)
                    count += 1
                else:
                    print(target_class_name, list(map(str, img_file_list)))

                # duration = time() - start_time
                # print(duration)
            print(count / nrof_images)


def get_embedding_and_path(line):
    """Calculate embedding of an image"""
    line = line.strip()
    emb_dict = json.loads(line)
    path, emb = list(emb_dict.items())[0]
    emb = np.array(emb)

    return emb, path


def get_class_name(path):
    return p.split(p.dirname(path))[-1]


def get_querie_class_name(path):
    dir_name = get_class_name(path)
    return dir_name.split(r'__')[0]


def insert_element(element, sorted_list, upper_bound=25):
    """Inserts an element in a sorted list"""
    assert len(sorted_list) <= upper_bound

    if sorted_list == []:
        return [element]

    result_list = sorted_list
    for i, e in enumerate(sorted_list):
        if element <= e:
            left_part = sorted_list[:i]
            right_part = sorted_list[i:]
            left_part.append(element)
            left_part.extend(right_part)
            result_list = left_part

            if len(result_list) > upper_bound:
                result_list.pop()

            break

    if len(result_list) < upper_bound:
        result_list.append(element)

    return result_list


class ImageFile:
    def __init__(self, path, score):
        self.path = path
        self.score = score

    def __le__(self, other):
        return self.score <= other.score

    def __str__(self):
        return get_querie_class_name(self.path) + ':' + str(self.score)


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

    parser.add_argument('--data_root', type=str,
                        help='Path to data directory which needs to be forward passed through the network',
                        default='../datasets/queries/')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf '
                             '(.pb) file',
                        default='../models/20190812-111258-gan-series-val78')
    parser.add_argument('--feature_vectors_file', type=str,
                        help='Path to the file with feature vectors',
                        default='feature_vectors_val78.txt')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.',
                        default=256)
    parser.add_argument('--top_n', type=int,
                        help='How many images to try to match',
                        default=5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


