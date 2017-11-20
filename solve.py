# -*- coding:utf-8 -*-
"""
@author:Luo
@file:solve.py
@time:2017/11/6 10:50
"""

import vgg
import layers
import tensorflow as tf
import numpy as np
import input_data
import math
import os
# import time

IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
MAX_STEP = 10000
NUM_TEST = 10000

def training():

    pretrained_weights = './pretrain/vgg16.npy'
    data_dir = './data/cifar10_data/cifar-10-batches-bin'
    train_log_dir = './log/train/'
    val_log_dir = './log/val/'

    with tf.name_scope('input'):
        images_train, labels_train = input_data.read_cifar10(data_dir, is_train=True,
                                                                        batch_size=BATCH_SIZE, shuffle=True)
        images_val, labels_val = input_data.read_cifar10(data_dir, is_train=False,
                                                                        batch_size=BATCH_SIZE, shuffle=False)

    image_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    label_holder = tf.placeholder(tf.int32, shape=[BATCH_SIZE, N_CLASSES])

    logits = vgg.VGG16(image_holder, N_CLASSES, 0.8)
    loss = layers.loss(logits, label_holder)
    accuracy = layers.accuracy(logits, label_holder)

    global_steps = tf.Variable(0, name='global_step', trainable=False)
    train_op = layers.optimize(loss, LEARNING_RATE, global_steps)

    saver = tf.train.Saver(tf.global_variables())

    # Refenrnce: https://stackoverflow.com/questions/35413618/tensorflow-placeholder-error-when-using-tf-merge-all-summaries
    summary_op = tf.summary.merge_all()
    # summary_op = tf.summary.merge([loss_summary, accuracy_summary], tf.GraphKeys.SUMMARIES)

    # The main thread
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    print('########################## Start Training ##########################')

    layers.load_with_skip(pretrained_weights, sess, ['fc6', 'fc7', 'fc8'])

    # Coordinate the relationship between threads
    # Reference: http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/threading_and_queues.html
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_summary_writer = tf.summary.FileWriter(train_log_dir, graph=sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, graph=sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            # start_time  = time .time()

            train_images, train_labels = sess.run([images_train, labels_train])
            _, train_loss, train_acc, summary_str = sess.run([train_op, loss, accuracy, summary_op],
                                                feed_dict={image_holder: train_images, label_holder: train_labels})
            # duration = time.time() - start_time

            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('step %d, loss = %.4f, accuracy = %.4f%%' % (step, train_loss, train_acc))
                #summary_str = sess.run(summary_op)
                train_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([images_val, labels_val])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={image_holder: val_images, label_holder: val_labels})
                print('step %d, val loss = %.2f, val accuracy = %.2f%%' % (step, val_loss, val_acc))

                #summary_str2 = sess.run(summary_op)
                val_summary_writer.add_summary(summary_str, step)

            # Why not use global_step=global_steps instead of step ???
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        coord.request_stop()

    coord.request_stop()
    coord.join(threads)

    sess.close()

def evaluate():

    with tf.Graph().as_default():
        log_dir = './log/train/'
        test_dir = './data/cifar10_data/cifar-10-batches-bin'

        test_images, test_labels = input_data.read_cifar10(test_dir, False,
                                                 BATCH_SIZE, False)
        logits = vgg.VGG16(test_images, 10, 1)
        correct = layers.correct_number(logits, test_labels)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found!')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('########################## Start Validation ##########################')

                print('\nEvaluating......')
                num_step = int(math.floor(NUM_TEST / BATCH_SIZE))
                num_sample = num_step * BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1

                # ******************************** Information of Testing ********************************
                print('Total testing samples: %d' % num_sample)
                print('Total correct predictions: %d' % total_correct)
                print('Average accuracy: %.2f%%' % (100 * total_correct / num_sample))

            except Exception as e:
                coord.request_stop(e)

            finally:
                coord.request_stop()
                coord.join(threads)

training()
# evaluate()


