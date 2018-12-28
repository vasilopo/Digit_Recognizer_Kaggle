from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


tf.logging.set_verbosity(tf.logging.INFO)


# Convolutional Neural Network Model
def cnn_model_deep_digits(features, labels, mode):

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten Layer
    flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer #1
    dense1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)

    # Dropout Layer for Dense #1
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Dense Layer #2
    dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense2, units=10)

    predictions = {

        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and
        # by the 'logging_hook'.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions= predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {

        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])

    }

    return tf.estimator.EstimatorSpec(mode=mode, loss= loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):

    # Load the dataset
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Split the dataset into x_train (input) and y_train (output)
    y_train = train['label']
    x_train = train.drop(labels=['label'], axis=1)

    # Normalize the Data
    x_train = x_train / 255.0
    test = test / 255.0

    # Reshape the Data from 1d array to 3d matrices
    x_train = x_train.values.reshape(-1, 28, 28, 1)
    #test = test.values.reshape(-1, 28, 28, 1)

    # Convert categorical values to OneHotArrays
    y_train = y_train.values.reshape(-1, 1)

    # Split the dataset into train and validation
    random_seed = 2
    train_data, eval_data, train_labels, eval_labels = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)

    # We save the model while training into the model_dir and from there we can start tensorboard by typing in terminal "tensorboard --logdir=/tmp/digit_classifier_model"
    digit_classifier = tf.estimator.Estimator(model_fn=cnn_model_deep_digits, model_dir="/tmp/digit_classifier_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=200)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    digit_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = digit_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()
