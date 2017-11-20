# -*- coding: utf-8 -*-
"""
Created on 2017/11/13 18:27

@author: Xiaoyang Wang
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import vgg
import time
import matplotlib.pyplot as plt

IMG_H = 32
IMG_W = 32

logs_train_dir = './log_dr5/train/'
file_path = './data/dr5_data/val/0/13_left.jpeg'
     
def inference(file_path):
    im = Image.open(file_path)
    image = im.resize([IMG_H, IMG_W], Image.ANTIALIAS)
    image_array = (np.array(image) - (255 / 2.0)) / 255
    # print(image_array.shape)

    with tf.Graph().as_default():
        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [1, IMG_H, IMG_W, 3])

        logit = vgg.VGG16(image, 5, 1)

        pred = tf.nn.softmax(logits=logit)
        
        x = tf.placeholder(tf.float32, shape=[IMG_H, IMG_W, 3])

        saver = tf.train.Saver()
        
        with tf.Session() as sess:

            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loading sucess,global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            start_time = time.time()
            prediction = sess.run(pred, feed_dict={x: image_array})
            duration = time.time() - start_time

            label = np.loadtxt('./dr_class.txt', str, delimiter='\t')
            index = np.argmax(prediction)
            grade = label[index]
            
            print('########################### Parameters ###########################')
            print('The label is:', grade)
            print('The probability is: ', np.max(prediction))
            print('Time used: %.4f' % (duration * 1000), 'ms')
            plt.imshow(im)
            plt.axis('off')
            plt.title('The grade of it is: %s' % index)
            plt.show()

inference(file_path)
