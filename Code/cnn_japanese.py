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

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

# The stub expects data in 64 * 64 * 1 (Single channel non RGB input)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 64, 64, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())
    # Convolutional Layer #2
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

    # dropout1 = tf.layers.dropout(
        # inputs=pool1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Convolutional Layer #3 and Pooling Layer #2
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # dropout2 = tf.layers.dropout(
        # inputs=pool2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=256, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())
    dropout3 = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=48, kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=48)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits)

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
            logits, name="softmax_tensor"),
        "labels_" : tf.argmax(
            input=labels, axis=1)
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    # Load training and eval data
    # mnist = learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    feature, labels = di.import_data('KATAKANA')

    feature_arr = np.asarray(feature, dtype = np.float32)
    labels_arr = np.asarray(labels)
    assert len(feature_arr) == len(labels_arr)
    p = np.random.permutation(len(feature_arr))
    feature_arr = feature_arr[p]
    labels_arr = labels_arr[p]
    labels_one_hot = pd.get_dummies(labels_arr).values
    # labels_arr = labels_arr - 9250.0
    #labels_arr = labels_arr - 166
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        feature_arr, labels_one_hot, test_size=0.10, random_state=42)

    # Create the Estimator
    mnist_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/japcnnn_cnn_model_kanji_god_4")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=100,
        monitors=[logging_hook])

# Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes", label_key="labels"),
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        x=eval_data, y=eval_labels, batch_size=100)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
