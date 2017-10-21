from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import dataimport as di
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
FLAGS = None

globalstep = 0


def weight_variable(name, shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
  initial = tf.get_variable(name, shape = shape , initializer=tf.contrib.layers.xavier_initializer())
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def next_batch(batch_size):
    global globalstep


def main(_):
    batch_size = 50
    # input handling section
    feature, labels = di.import_data('KATAKANA')

    feature_arr = np.asarray(feature, dtype=np.float32)
    labels_arr = np.asarray(labels)
    assert len(feature_arr) == len(labels_arr)
    perm = np.random.permutation(len(feature_arr))
    feature_arr = feature_arr[perm]
    labels_arr = labels_arr[perm]
    #labels_arr = labels_arr - 9250.0
    labels_arr = labels_arr - 166
    # labels_one_hot = tf.one_hot(labels_arr, depth = 48, on_value = "1", off_value= "0", axis = -1)
    # labels_one_hot = np.eye(48)[labels_arr]
    labels_one_hot = pd.get_dummies(labels_arr).values
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        feature_arr, labels_one_hot, test_size=0.10, random_state=42)

    x = tf.placeholder(tf.float32, shape=[None, 4096])
    y_ = tf.placeholder(tf.float32, shape=[None, 48])

    # with tf.variable_scope('conv1'):
    # W_conv1 = tf.get_variable('W_conv1', shape=[3, 3, 1, 32], initializer=tf.truncated_normal_initializer())
    W_conv1 = tf.get_variable("W_conv1", shape=[3, 3, 1, 32],
           initializer=tf.contrib.layers.xavier_initializer())
    # W_conv1 = weight_variable('W_conv1', [3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 64, 64, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # with tf.variable_scope('conv2'):
    # W_conv2 = tf.get_variable('W_conv2', shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer())
    W_conv2 = tf.get_variable("W_conv2", shape=[3, 3, 32, 64],
        initializer=tf.contrib.layers.xavier_initializer())
    # W_conv2 = weight_variable('W_conv2', [3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # with tf.variable_scope('fc1'):
    # W_fc1 = tf.get_variable('w', shape=[16 * 16 * 64, 256], initializer=tf.truncated_normal_initializer())
    W_fc1 = tf.get_variable("W_fc1", shape=[16 * 16 * 64, 256],
        initializer=tf.contrib.layers.xavier_initializer())
    # W_fc1 = weight_variable('W_fc1', [16 * 16 * 64, 256])
    b_fc1 = bias_variable([256])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # with tf.variable_scope('fc2'):
    # W_fc2 = tf.get_variable('w', shape=[256, 48], initializer=tf.truncated_normal_initializer())
    W_fc2 = tf.get_variable("W_fc2", shape= [256, 48],
        initializer=tf.contrib.layers.xavier_initializer())
    # W_fc2 = weight_variable('W_fc2', [256, 48])
    b_fc2 = bias_variable([48])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #onehot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=48)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        # batch = next_batch(50)
        input_batch, label_batch = train_data[i * batch_size:(
            i + 1) * batch_size], train_labels[i * batch_size:(i + 1) * batch_size]
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: input_batch, y_: label_batch, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: input_batch, y_: label_batch, keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: eval_data, y_: eval_labels, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
