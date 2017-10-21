# Instructions on how to train the network and obtain the results
#
# Select the network to be used : M6_1 or M8
# Set this value in the model_fn parameter of the estimator
#
#    mnist_classifier = learn.Estimator(
#        model_fn=M8, model_dir=log_path + "/japcnn_Kanji_god_M8_try2")
#
# Select the script to be trained on :
# Distinct label counts for script types
#
#
# Script type   Distinct Classes
# -----------   ----------------
# HIRAGANA      75
# KATAKANA      48
# KANJI         881
# ALL           1004

# Change the corresponding class values in
# onehot_labels = tf.one_hot(
#       indices=tf.cast(labels, tf.int32), depth=881)
# and..
#
# logits = tf.layers.dense(inputs=dropout3, units=881, kernel_initializer=tf.contrib.layers.xavier_initializer(),
#       bias_initializer=tf.zeros_initializer())
#
# Directory structure expected:
# Code
# |----jp_recog_main.py
# |----dataimport.py
# dataset
# |----ETL1
#       |-----ETL1C_01
                # .
                # .
                # .
#       |-----ETL1C_13
# |----ETL8B
#       |-----ETL8B2C1
                # .
                # .
                # .
#        |-----ETL8B2C3


# Author    : Rohin Mohanadas
# Date      : 22/05/2017
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import dataimport as di
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import os
import platform

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

# The stub expects data in 64 * 64 * 1 (Single channel non RGB input)
def M6_1(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 64, 64, 1])

    # Convolutional Layer #1 ( weights initialized with xavier initializer )
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())
    # Convolutional Layer #2 ( weights initialized with xavier initializer )
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    dropout1 = tf.layers.dropout(
        inputs=pool1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Convolutional Layer #3  ( weights initialized with xavier initializer )
    conv3 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    # Convolutional Layer #4 and Pooling Layer #2  ( weights initialized with
    # xavier initializer )
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    dropout2 = tf.layers.dropout(
        inputs=pool2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Dense Layer
    pool2_flat = tf.reshape(dropout2, [-1, 16 * 16 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=256, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())
    dropout3 = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer ( weights initialized with xavier initializer )
    logits = tf.layers.dense(inputs=dropout3, units=881, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.zeros_initializer())

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(
            indices=tf.cast(labels, tf.int32), depth=881)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=1e-4,
            optimizer='Adam')

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def M8(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 64, 64, 1])

    # Convolutional Layer #1 ( weights initialized with xavier initializer )
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())
    # Convolutional Layer #2 ( weights initialized with xavier initializer )
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    dropout1 = tf.layers.dropout(
        inputs=pool1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Convolutional Layer #3  ( weights initialized with xavier initializer )
    conv3 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    # Convolutional Layer #4 and Pooling Layer #2  ( weights initialized with
    # xavier initializer )
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    dropout2 = tf.layers.dropout(
        inputs=pool2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Dense Layer
    conv5 = tf.layers.conv2d(
        inputs=dropout2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

    dropout3 = tf.layers.dropout(
        inputs=pool3, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    pool3_flat = tf.reshape(dropout3, [-1, 8 * 8 * 128])
    dense = tf.layers.dense(
        inputs=pool3_flat, units=1024, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    dropout4 = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer ( weights initialized with xavier initializer )
    logits = tf.layers.dense(inputs=dropout4, units=881, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.zeros_initializer())

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(
            indices=tf.cast(labels, tf.int32), depth=881)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=1e-4,
            optimizer='Adam')

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)



def main(unused_argv):

    # Distinct label counts for script types 
    # HIRAGANA  75
    # KATAKANA  48
    # KANJI     881
    # ALL       1004

    # Load training and eval data
    feature, labels = di.import_data('KANJI')

    # Logging folder and model save location
    if platform.system() in ["Darwin", "Linux"]:
        log_path = os.path.dirname(os.path.dirname(
            (os.path.abspath(__file__)))) + '/logs'
    else:
        log_path = os.path.dirname(os.path.dirname(
            (os.path.abspath(__file__)))) + '\\logs'

    feature_arr = np.asarray(feature, dtype=np.float32)
    labels_arr = np.asarray(labels)

    # test to ensure feature and label compatibility
    assert len(feature_arr) == len(labels_arr)

    # permute the otherwise ordered dataset for better training progress
    p = np.random.permutation(len(feature_arr))
    feature_arr = feature_arr[p]
    labels_arr = labels_arr[p]

    # switch from the japanese character code to unique label list for use in
    # tensorflow
    unique_labels = list(set(labels_arr))
    labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    new_labels = np.array([labels_dict[l] for l in labels_arr], dtype=np.int32)

    # split training and test data
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        feature_arr, new_labels, test_size=0.30, random_state=42)

    validation_metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes")
    }

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        x=train_data,
        y=train_labels,
        every_n_steps=100,
        metrics=validation_metrics)

    # Create the Estimator, log path is where the models get stored Change this value on
    # successive runs if the models are to be preserved.

    # Change the model function to select M6_1 or M8
    mnist_classifier = learn.Estimator(
        model_fn=M8, model_dir=log_path + "/japcnn_Kanji_god_M8_try2")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=200)

    # Train the model - batch size of 100 and step count of 10000
    mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=10000,
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        x=eval_data, y=eval_labels, batch_size=100, metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
