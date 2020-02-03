"""Uses trained model to calculate embeddings and saves them in a feature_vectors.txt file"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
import argparse
import facenet
import json

from smaug.data import ImageData
from geo_utils import coordinates_from_file


def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch_res_p:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load data
            dataset = facenet.get_dataset(args.data_root)
            image_paths, _ = facenet.get_image_paths_and_labels(dataset)
            images = ImageData(image_paths, sess, load_size=args.image_size, batch_size=1)
            nrof_images = len(images)

            with open("feature_vectors.txt", "w") as file:

                for i in range(nrof_images):
                    img = images.batch()
                    path = os.path.abspath(image_paths[i])

                    # Run forward pass to calculate embeddings
                    feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)

                    # Get gps coordinates
                    try:
                        coordinates = coordinates_from_file(path)
                    except:
                        coordinates = None
                    line_dict = {path: emb[0].tolist(), "coordinates": coordinates}
                    json_data = json.dumps(line_dict)

                    file.write(json_data)
                    file.write('\n')
                    print(path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str,
                        help='Path to data directory which needs to forward passed through the network',
                        default='../datasets/series_stash/')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf '
                             '(.pb) file',
                        default='../models/test_pb/wout_phase.pb')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.',
                        default=256)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
