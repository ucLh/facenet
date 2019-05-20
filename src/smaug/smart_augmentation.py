from functools import partial

import tensorflow as tf
import tensorflow.contrib.layers as layers

initializer = tf.truncated_normal_initializer(stddev=0.02)
relu = tf.nn.relu
conv_relu = partial(layers.conv2d, activation_fn=relu, weights_initializer=initializer, biases_initializer=None)
conv = partial(layers.conv2d, activation_fn=None, weights_initializer=initializer, biases_initializer=None)


def smart_augmentation(img, channels=6, scope='Smart_Augmentation'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = conv_relu(img, 16, 3)
        net = conv_relu(net, 16, 5)
        net = conv_relu(net, 32, 7)
        net = conv_relu(net, 32, 5)
        net = conv(net, channels // 2, 1, activation_fn=tf.nn.tanh)

        return net
