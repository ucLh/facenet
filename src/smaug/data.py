from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from collections import deque

import numpy as np
import tensorflow as tf

from facenet import get_image_paths_and_labels
from augmentations.augmentations import random_black_patches


class ImageDataRaw:

    def __init__(self,
                 image_paths,
                 session,
                 batch_size=10,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=False,
                 buffer_size=2048,
                 repeat=1):
        self._sess = session
        self.image_paths = image_paths
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._drop_remainder = drop_remainder
        self._num_threads = num_threads
        self._shuffle = shuffle
        self._buffer_size = buffer_size
        self._repeat = repeat
        self._img_batch = self._image_batch().make_one_shot_iterator().get_next()
        self._img_num = len(image_paths)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    def _parse_func(self, path):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self._num_channels)
        return img

    def _map_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        return dataset.map(self._parse_func, num_parallel_calls=self._num_threads)

    def _image_batch(self):
        dataset = self._map_dataset()

        if self._shuffle:
            dataset = dataset.shuffle(self._buffer_size)

        dataset = dataset.batch(self._batch_size, drop_remainder=self._drop_remainder)
        dataset = dataset.repeat(self._repeat).prefetch(2)

        return dataset


class ImageData:

    def __init__(self,
                 image_paths,
                 session,
                 batch_size=10,
                 load_size=256,
                 use_black_patches=False,
                 use_crop=False,
                 crop_size=256,
                 use_flip=False,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=8,
                 shuffle=False,
                 buffer_size=2048,
                 repeat=1):
        self._sess = session
        self.image_paths = image_paths
        self._batch_size = batch_size
        self._load_size = load_size
        self._use_black_patches = use_black_patches
        self._use_crop = use_crop
        self._crop_size = crop_size
        assert load_size >= crop_size
        self._use_flip = use_flip
        self._num_channels = num_channels
        self._drop_remainder = drop_remainder
        self._num_threads = num_threads
        self._shuffle = shuffle
        self._buffer_size = buffer_size
        self._repeat = repeat
        self._img_batch = self._image_batch().make_one_shot_iterator().get_next()
        self._img_num = len(image_paths)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    def _parse_func(self, path):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self._num_channels)
        # if self._use_flip:
        #     img = tf.image.random_flip_left_right(img)
        #
        # # img = tf.compat.v2.image.resize(img, [self._load_size, self._load_size])
        # img = tf.image.resize_images(img, [self._load_size, self._load_size])
        # img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        #
        # if self._use_black_patches:
        #     img = random_black_patches(img)
        #
        # if self._use_crop:
        #     img = tf.random_crop(img, [self._crop_size, self._crop_size, self._num_channels])
        # img = img * 2 - 1
        return img

    def _map_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        return dataset.map(self._parse_func, num_parallel_calls=self._num_threads)

    def _image_batch(self):
        dataset = self._map_dataset()

        if self._shuffle:
            dataset = dataset.shuffle(self._buffer_size)

        dataset = dataset.batch(self._batch_size, drop_remainder=self._drop_remainder)
        dataset = dataset.repeat(self._repeat).prefetch(2)

        return dataset

    @staticmethod
    def resize_images_w_cv(filenames, load_size):
        result = []
        for filename in filenames:
            file_contents = cv2.imread(filename)
            image = cv2.resize(file_contents, (load_size, load_size), cv2.INTER_LINEAR)
            # image = tf.convert_to_tensor(image, dtype=tf.float32)
            result.append(image)
        result = np.array(result)
        return result


class LabeledImageData(ImageData):

    def __init__(self,
                 image_paths,
                 labels,
                 session,
                 batch_size=1,
                 load_size=256,
                 use_black_patches=False,
                 use_crop=True,
                 crop_size=256,
                 use_flip=True,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=True,
                 buffer_size=2048,
                 repeat=-1):

        self.labels = labels

        super().__init__(image_paths=image_paths,
                         session=session,
                         batch_size=batch_size,
                         load_size=load_size,
                         use_black_patches=use_black_patches,
                         use_crop=use_crop,
                         crop_size=crop_size,
                         use_flip=use_flip,
                         num_channels=num_channels,
                         drop_remainder=drop_remainder,
                         num_threads=num_threads,
                         shuffle=shuffle,
                         buffer_size=buffer_size,
                         repeat=repeat)

    def _parse_func(self, path):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self._num_channels)
        return img

    def _map_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        dataset = dataset.map(self._parse_func, num_parallel_calls=self._num_threads)

        return dataset.zip((dataset, labels))


class SmaugImageData(ImageData):

    def __init__(self,
                 image_classes,
                 flat_paths,
                 pair_dataset_name,
                 session,
                 batch_size=1,
                 load_size=286,
                 use_crop=True,
                 crop_size=256,
                 use_flip=True,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=True,
                 buffer_size=2048,
                 repeat=-1):
        """
        :param image_classes: A list of ImageClass objects defined in src/facenet.py
        :param flat_paths: Flat image paths
        :param pair_dataset_name: A name of the dataset with pair images. This dataset has to have the same structure as
               the original one (see --data_dir parameter in train_softmax_w_smaug.py)
        """
        self._image_classes = image_classes
        self._pair_dataset_name = pair_dataset_name
        self.pair_paths = create_pair_paths(flat_paths, pair_dataset_name)
        self.label_paths = self._create_label_paths(image_classes, pair_dataset_name)
        _, self.facenet_labels = get_image_paths_and_labels(image_classes)
        self._img_num = len(flat_paths*2)

        super().__init__(image_paths=flat_paths,
                         session=session,
                         batch_size=batch_size,
                         load_size=load_size,
                         use_crop=use_crop,
                         crop_size=crop_size,
                         use_flip=use_flip,
                         num_channels=num_channels,
                         drop_remainder=drop_remainder,
                         num_threads=num_threads,
                         shuffle=shuffle,
                         buffer_size=buffer_size,
                         repeat=repeat)

    @staticmethod
    def _create_label_paths(image_classes, pair_dataset_name):
        label_paths = []
        for img_class in image_classes:
            size = len(img_class)
            paths = deque(img_class.image_paths)
            if size-1 > 0:
                ind = random.randint(1, size-1)
            else:
                ind = 0
            paths.rotate(ind)
            for i, path in enumerate(paths):
                class_switch = random.randint(1, 2)
                if class_switch == 1:
                    paths[i] = _change_dataset_name_in_path(path, pair_dataset_name)

            label_paths += list(paths)
        return label_paths

    def _map_dataset(self):
        dataset_a = tf.data.Dataset.from_tensor_slices(self.image_paths)
        dataset_b = tf.data.Dataset.from_tensor_slices(self.pair_paths)
        labels = tf.data.Dataset.from_tensor_slices(self.label_paths)
        facenet_labels = tf.data.Dataset.from_tensor_slices(self.facenet_labels)

        dataset_a = dataset_a.map(self._parse_func, num_parallel_calls=self._num_threads)
        dataset_b = dataset_b.map(self._parse_func, num_parallel_calls=self._num_threads)
        labels = labels.map(self._parse_func, num_parallel_calls=self._num_threads)

        return tf.data.Dataset.zip((dataset_a, dataset_b, labels, facenet_labels))


def _change_dataset_name_in_path(path, new_name):
    new_path_list = path.split('/')

    # change dataset name
    new_path_list[-3] = new_name
    return '/'.join(new_path_list)


def create_pair_paths(image_paths, pair_dataset_name):
    pair_paths = []
    for path in image_paths:
        new_path = _change_dataset_name_in_path(path, pair_dataset_name)
        pair_paths.append(new_path)
    return pair_paths
