from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from augmentations.augmentations import random_black_patches
from collections import deque
from facenet import get_image_paths_and_labels
import tensorflow as tf
import random


class ImageData:

    def __init__(self,
                 image_paths,
                 labels,
                 session,
                 batch_size=1,
                 load_size=286,
                 crop_size=256,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=True,
                 buffer_size=2048):
        self._sess = session
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.load_size = load_size
        self.crop_size = crop_size
        self.num_channels = num_channels
        self.drop_remainder = drop_remainder
        self.num_threads = num_threads
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self._img_batch = self._image_batch().make_one_shot_iterator().get_next()
        self._img_num = len(image_paths)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    def _parse_func(self, path):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self.num_channels)
        # img = tf.image.random_flip_left_right(img)
        img = tf.image.resize_images(img, [self.load_size, self.load_size])
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        # img = img - tf.reduce_mean(img)
        # img = tf.random_crop(img, [self.crop_size, self.crop_size, self.num_channels])
        img = img * 2 - 1
        return img

    def _image_batch(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        dataset = dataset.map(self._parse_func, num_parallel_calls=self.num_threads)

        dataset = dataset.zip((dataset, labels))
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)

        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.repeat(-1).prefetch(2)

        return dataset


class PairedImageData(ImageData):

    def __init__(self,
                 image_paths_a,
                 image_paths_b,
                 session,
                 batch_size=1,
                 load_size=286,
                 crop_size=256,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=True,
                 buffer_size=4096):
        super().__init__([],
                         session,
                         batch_size,
                         load_size,
                         crop_size,
                         num_channels,
                         drop_remainder,
                         num_threads,
                         shuffle,
                         buffer_size)
        self.image_paths_a = image_paths_a
        self.image_paths_b = image_paths_b
        self._img_num = len(image_paths_a) + len(image_paths_b)

    def _image_batch(self):
        dataset_a = tf.data.Dataset.from_tensor_slices(self.image_paths_a)
        dataset_b = tf.data.Dataset.from_tensor_slices(self.image_paths_b)

        dataset_a = dataset_a.map(self._parse_func, num_parallel_calls=self.num_threads)
        dataset_b = dataset_b.map(self._parse_func, num_parallel_calls=self.num_threads)

        dataset = tf.data.Dataset.zip((dataset_a, dataset_b))

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)

        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)

        return dataset


class SmaugImageData:

    def __init__(self,
                 image_classes,
                 flat_paths,
                 pair_dataset_name,
                 session,
                 batch_size=1,
                 load_size=286,
                 crop_size=256,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=True,
                 buffer_size=2048):
        self._sess = session
        self.image_classes = image_classes
        self.image_paths = flat_paths
        self.pair_dataset_name = pair_dataset_name
        self.batch_size = batch_size
        assert load_size >= crop_size
        self.load_size = load_size
        self.crop_size = crop_size
        self.num_channels = num_channels
        self.drop_remainder = drop_remainder
        self.num_threads = num_threads
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.pair_paths = self._create_pair_paths(flat_paths, pair_dataset_name)
        self.label_paths = []
        self._create_label_paths()
        _, self.facenet_labels = get_image_paths_and_labels(image_classes)
        self._img_batch = self._image_batch().make_one_shot_iterator().get_next()
        self._img_num = len(image_classes)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    def _parse_func(self, path):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self.num_channels)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.resize_images(img, [self.load_size, self.load_size])
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        # img = img - tf.reduce_mean(img)
        img = tf.random_crop(img, [self.crop_size, self.crop_size, self.num_channels])
        img = random_black_patches(img)
        img = img * 2 - 1
        return img

    @staticmethod
    def _change_dataset_name_in_path(path, new_name):
        new_path_list = path.split('/')

        # change dataset name
        new_path_list[-3] = new_name
        return '/'.join(new_path_list)

    @staticmethod
    def _create_pair_paths(image_paths, pair_dataset_name):
        pair_paths = []
        for path in image_paths:
            new_path = SmaugImageData._change_dataset_name_in_path(path, pair_dataset_name)
            pair_paths.append(new_path)
        return pair_paths

    def _create_label_paths(self):
        for img_class in self.image_classes:
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
                    paths[i] = self._change_dataset_name_in_path(path, self.pair_dataset_name)

            self.label_paths += list(paths)

    def _image_batch(self):
        dataset_a = tf.data.Dataset.from_tensor_slices(self.image_paths)
        dataset_b = tf.data.Dataset.from_tensor_slices(self.pair_paths)
        labels = tf.data.Dataset.from_tensor_slices(self.label_paths)
        facenet_labels = tf.data.Dataset.from_tensor_slices(self.facenet_labels)

        dataset_a = dataset_a.map(self._parse_func, num_parallel_calls=self.num_threads)
        dataset_b = dataset_b.map(self._parse_func, num_parallel_calls=self.num_threads)
        labels = labels.map(self._parse_func, num_parallel_calls=self.num_threads)

        dataset = tf.data.Dataset.zip((dataset_a, dataset_b, labels, facenet_labels))

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)

        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.repeat(-1).prefetch(2)

        return dataset
