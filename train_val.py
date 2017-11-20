# -*- coding:utf-8 -*-
"""
@author:Luo
@file:train_val.py
@time:2017/11/10 15:53
"""

import vgg
import layers
import tensorflow as tf
import numpy as np
import dr5_input
import math
import os
# import time

IMG_W = 32
IMG_H = 32
N_CLASSES = 5
BATCH_SIZE = 128
MAX_STEP = 40000
NUM_TEST = 10000

start_rate = 1e-2
decay_steps = 30000
deacy_rate = 0.96

def training():
    pretrained_weights = './pretrain/vgg16.npy'
    
    train_log_dir = './log_dr50000/train/'
    val_log_dir = './log_dr50000/val/'

    with tf.name_scope('input'):
        images_train, labels_train = dr5_input.input_data(True, BATCH_SIZE)
        images_val, labels_val = dr5_input.input_data(False, BATCH_SIZE)

    image_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    label_holder = tf.placeholder(tf.int32, shape=[BATCH_SIZE, N_CLASSES])

    logits = vgg.VGG16(image_holder, N_CLASSES, 0.5)
    loss = layers.loss(logits, label_holder)
    accuracy = layers.accuracy(logits, label_holder)

    global_steps = tf.Variable(0, name='global_step', trainable=False)
    LEARNING_RATE=tf.train.exponential_decay(start_rate,
                                            global_steps,
                                            decay_steps,
                                            deacy_rate,
                                            staircase=True)
    train_op = layers.optimize(loss, LEARNING_RATE, global_steps)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    # The main thread
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    sess = tf.InteractiveSession()
    sess.run(init)

    print('########################## Start Training ##########################')

    layers.load_with_skip(pretrained_weights, sess, ['fc6', 'fc7', 'fc8'])

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
                train_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([images_val, labels_val])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={image_holder: val_images, label_holder: val_labels})
                print('step %d, val loss = %.2f, val accuracy = %.2f%%' % (step, val_loss, val_acc))
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                lr = sess.run(LEARNING_RATE)  
                print ("step %d, learning_rate= %f"% (step,lr)) 
                 

    except tf.errors.OutOfRangeError:
        coord.request_stop()

    coord.request_stop()
    coord.join(threads)

    sess.close()

def evaluate():
    with tf.Graph().as_default():
        log_dir = './log_dr5/train/'
#         test_dir = './data/cifar10_data/cifar-10-batches-bin'
#
#         test_images, test_labels = input_data.read_cifar10(test_dir, False,
#                                                  BATCH_SIZE, False)
        images_val, labels_val = dr5_input.input_data(False, BATCH_SIZE)
       
        logits = vgg.VGG16(images_val, N_CLASSES, 1)
        correct = layers.correct_number(logits, labels_val)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt:
                sub_ckpt = ckpt.all_model_checkpoint_paths[1]
                global_step = sub_ckpt.split('/')[-1].split('-')[-1]
                saver.restore(sess, sub_ckpt)
                print('Loading success, global_step is %s' % global_step)
            
            #if ckpt and ckpt.model_checkpoint_path:
            #   global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            #   saver.restore(sess, ckpt.model_checkpoint_path)
            #   print('Loading success, global_step is %s' % global_step)
           
            else:
                print('No checkpoint file found!')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('########################## Start Validation ##########################')

                print('\nEvaluating......')
                num_step = int(math.floor(NUM_TEST / BATCH_SIZE)) + 1
                # num_step = math.ceil(NUM_TEST / BATCH_SIZE)
                # num_sample = num_step * BATCH_SIZE
                num_sample = NUM_TEST
                
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

#training()
evaluate()