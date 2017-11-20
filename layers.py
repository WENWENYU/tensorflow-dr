import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

def conv(layer_name, x, out_channels, kernel_size, stride, is_pretrain=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                                  trainable=is_pretrain,
                                  shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                  initializer=xavier_initializer())
        b = tf.get_variable(name='biases',
                               trainable=is_pretrain,
                               shape=[out_channels],
                               initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')

        return x

def pool(layer_name, x, kernel, stride, is_max_pool=True):
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)

    return x


def fc_layer(layer_name, x, out_channels):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_channels],
                            initializer=xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)

        return x

def dropout(layer_name, x, keep_prob=0.5):
    x = tf.nn.dropout(x, keep_prob=keep_prob, name=layer_name)

    return x

def correct_number(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    n_correct = tf.cast(correct, tf.int32)

    return tf.reduce_sum(n_correct)

def loss(logits, labels):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        # tf.summary.scalar(scope+'/loss', loss)

        return loss

def accuracy(logits, labels):
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0

        # Why not out of service ???
        # tf.summary.scalar(scope+'/accuracy', accuracy)

        return accuracy

def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op


######################### Only for test #########################
def load_with_test(data_path):
    data_dict= np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print(key)
        print('weights.shape:', weights.shape)
        print('biases.shape:', biases.shape)
######################### Only for test #########################

def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))

# load_with_test('./pretrain/vgg16.npy')








