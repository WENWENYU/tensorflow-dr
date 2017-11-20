# -*- coding:utf-8 -*-
"""
@author:Luo
@file:dr5_input.py
@time:2017/11/8 10:22
"""

import tensorflow as tf
import numpy as np
import os
# import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image

train_dir = './data/dr5_data/train/'
val_dir = './data/dr5_data/val/'

# data_dir = './data/dr5_data/'
save_dir = './tfRecords/'
train_tfrecords = 'train.tfrecords'
val_tfrecords = 'val.tfrecords'
img_resize = 32
total = 8

def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for file in files:
            images.append(os.path.join(root, file))
            # img = io.imread(os.path.join(root, file))
            # images.append(img)

        # get 10 sub_folder names
        for folder in sub_folders:
            temp.append(os.path.join(root, folder))

    labels = []
    for folder in temp:
        n_img = len(os.listdir(folder))
        num = folder.split('/')[-1]
        print('Numbers of class %d is: ' % int(num), n_img)

        if num == '0':
            labels = np.append(labels, n_img * [0])
        elif num == '1':
            labels = np.append(labels, n_img * [1])
        elif num == '2':
            labels = np.append(labels, n_img * [2])
        elif num == '3':
            labels = np.append(labels, n_img * [3])
        else:
            labels = np.append(labels, n_img * [4])

    # shuffle
    temp = np.array([images, labels])

    # Only for test
    # print(np.where(labels == 4.))

    temp = np.transpose(temp)
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

# Coding function
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# For more information refer to tensorflow book on page 88
def convert_to(images, labels, save_dir, tfrecords):
    filename = os.path.join(save_dir, tfrecords)
    n_samples = len(labels)

    if len(images) != n_samples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

    # Reference: http://ourcoders.com/thread/show/8659/
    # rows = images.shape[1]
    # cols = images.shape[2]
    # depth = images.shape[3]

    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for index in np.arange(0, n_samples):
        try:
            print('Index of images is: ', index)
            # image = io.imread(images[index])  # type(image) must be array!
            # rows = image.shape[0]
            # cols = image.shape[1]
            # depth = image.shape[2]
            # image_raw = image.tostring()
            im = Image.open(images[index])
            im = im.resize((img_resize, img_resize), Image.ANTIALIAS)
            image_raw = im.tobytes()
            label = int(labels[index])
            example = tf.train.Example(features=tf.train.Features(feature={
				# 'height': _int64_feature(rows),
				# 'width': _int64_feature(cols),
				# 'depth': _int64_feature(depth),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[index])
            print('error: %s' % e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
			# 'height': tf.FixedLenFeature([], tf.int64),
			# 'width': tf.FixedLenFeature([], tf.int64),
			# 'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    # image_height = tf.cast(features['height'], tf.int32)
    # image_width = tf.cast(features['width'], tf.int32)
    # image_channels = tf.cast(features['depth'], tf.int32)

    record_image = tf.decode_raw(features['image_raw'], tf.uint8)

    image = tf.reshape(record_image, tf.stack([img_resize, img_resize, 3]))

    # -0.5~0.5
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
  
    label = tf.cast(features['label'], tf.int32)

    return image, label

def input_data(is_train, batch_size):
    # is_train=True for train data else for validation data
    filename = os.path.join(save_dir, train_tfrecords if is_train else val_tfrecords)
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([filename])
    image, label = read_and_decode(filename_queue)

    min_after_dequeue = 10000
    # image_size = 512
    num_threads = 64
    # image = tf.image.resize_images(image, size=[image_size, image_size])

    # Usually capacity = min_after_dequeue + 3 * batch_size !!!
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=num_threads,
                                                      min_after_dequeue=min_after_dequeue,
                                                      capacity=capacity)

    # one-hot
    n_classes = 5
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch

def plot_images(images, labels, total=20):
    if total % 2 != 0:
        raise Exception('The number of pictures entered should be a multiple of 2...')
    for i in np.arange(total):
        plt.subplot(2, total/2, i + 1)
        plt.axis('off')
        plt.title(str(labels[i]), fontsize=14)
        plt.subplots_adjust(top=1)
        plt.imshow(images[i])
    plt.show()
	
############################ Wrapper ############################
def images_to_tfrecords(data_dir, tfrecords):
    image_list, label_list = get_file(data_dir)
    convert_to(image_list, label_list, save_dir, tfrecords)
############################ Wrapper ############################
def main():
    with tf.Graph().as_default():
        images, labels = input_data(is_train=False, batch_size=8)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init_op)

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Permanent cycle mode
            while not coord.should_stop():
                t_images, t_labels = sess.run([images, labels])
                plot_images(t_images, t_labels, 8)
                return
        except tf.errors.OutOfRangeError:
            print('done!')

        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

def test():
    filename_queue = tf.train.string_input_producer(['tfRecords/' + val_tfrecords])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [img_resize, img_resize, 3])

    label = tf.cast(features['label'], tf.int32)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        t_image, t_label = sess.run([image, label])
        im = Image.fromarray(t_image, 'RGB')
        im.show()
        coord.request_stop()
        coord.join(threads)

################################## Only for test ##################################
# image_list, label_list = get_file('./data/dr5_data/')
# img = cv2.imread(image_list[0])
# cv2.imshow('DR', img)
# cv2.waitKey(0)
################################## Only for test ##################################	

# main()

# test()

# images_to_tfrecords(train_dir, train_tfrecords)
# images_to_tfrecords(val_dir, val_tfrecords)
