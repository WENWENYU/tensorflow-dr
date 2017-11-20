# -*- coding:utf-8 -*-
"""
@author:Luo
@file:vgg.py.py
@time:2017/11/6 10:34
"""
import tensorflow as tf
import layers

def VGG16(x, n_classes, keep_prob):
    with tf.name_scope('VGG16'):
        # Group 1
        x = layers.conv('conv1_1', x, 64, [3, 3], [1, 1, 1, 1])
        x = layers.conv('conv1_2', x, 64, [3, 3], [1, 1, 1, 1])
        with tf.name_scope('pool1'):
            x = layers.pool('pool1', x, [1, 2, 2, 1], [1, 2, 2, 1])

        # Group 2
        x = layers.conv('conv2_1', x, 128, [3, 3], [1, 1, 1, 1])
        x = layers.conv('conv2_2', x, 128, [3, 3], [1, 1, 1, 1])
        with tf.name_scope('pool2'):
            x = layers.pool('pool2', x, [1, 2, 2, 1], [1, 2, 2, 1])

        # Group 3
        x = layers.conv('conv3_1', x, 256, [3, 3], [1, 1, 1, 1])
        x = layers.conv('conv3_2', x, 256, [3, 3], [1, 1, 1, 1])
        x = layers.conv('conv3_3', x, 256, [3, 3], [1, 1, 1, 1])
        with tf.name_scope('pool3'):
            x = layers.pool('pool3', x, [1, 2, 2, 1], [1, 2, 2, 1])

        # Group 4
        x = layers.conv('conv4_1', x, 512, [3, 3], [1, 1, 1, 1])
        x = layers.conv('conv4_2', x, 512, [3, 3], [1, 1, 1, 1])
        x = layers.conv('conv4_3', x, 512, [3, 3], [1, 1, 1, 1])
        with tf.name_scope('pool4'):
            x = layers.pool('pool4', x, [1, 2, 2, 1], [1, 2, 2, 1])

        # Group 5
        x = layers.conv('conv5_1', x, 512, [3, 3], [1, 1, 1, 1])
        x = layers.conv('conv5_2', x, 512, [3, 3], [1, 1, 1, 1])
        x = layers.conv('conv5_3', x, 512, [3, 3], [1, 1, 1, 1])
        with tf.name_scope('pool5'):
            x = layers.pool('pool5', x, [1, 2, 2, 1], [1, 2, 2, 1])

        x = layers.fc_layer('fc6', x, 4096)
        x = layers.dropout('drop6', x, keep_prob)

        x = layers.fc_layer('fc7', x, 4096)
        x = layers.dropout('drop7', x, keep_prob)

        x = layers.fc_layer('fc8', x, n_classes)

        return x
