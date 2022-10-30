from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from conv_model import CustomSequential

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


###############################################################################################


def get_data():
    """
    Loads CIFAR10 training and testing datasets

    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
            D0: TF Dataset training subset
            D1: TF Dataset testing subset
        D_info: TF Dataset metadata
    """

    import tensorflow_datasets as tfds

    ## Overview of dataset downloading: https://www.tensorflow.org/datasets/catalog/overview
    ## CIFAR-10 Dataset https://www.tensorflow.org/datasets/catalog/cifar10
    (D0, D1), D_info = tfds.load(
        "cifar10", as_supervised=True, split=["train[:50%]", "test"], with_info=True
    )

    X0, X1 = [np.array([r[0] for r in tfds.as_numpy(D)]) for D in (D0, D1)]
    Y0, Y1 = [np.array([r[1] for r in tfds.as_numpy(D)]) for D in (D0, D1)]

    return X0, Y0, X1, Y1, D0, D1, D_info


###############################################################################################


def run_task(data, task, subtask="all", epochs=None, batch_size=None):
    """
    Runs model on a given dataset.

    :param data: Input dataset to train on
    :param task: 1 => train the model with tf.keras.layers (tf.keras.layers.Conv2D,
                      tf.keras.layers.Dense, etc.). Only use this to find a good model. This
                      will NOT be enabled on the autograder.
                 2 => train the model with some/all of the layers
                      you've implemented in layers_keras.py (which layers to substitute are
                      specified in subtask)
                 3 => train the model with your manual implementation of Conv2D
    :param subtask: 1 => train the model with the Conv2D layer you've implemented in layers_keras.py
                    2 => train the model with the BatchNormalization layer you've implemented in
                         layers_keras.py
                    3 => train the model with the Dropout layer you've implemented in
                         layers_keras.py
                    all => use all layers from layers_keras.py

    :return trained model
    """
    import conv_model  ## Where model, preprocessing, and augmentation pipelines are.
    import layers_keras  ## Where layer subclass components are
    import layers_manual  ## Where manual non-diffable conv implementation resides

    ## Retrieve data from tuple
    X0, Y0, X1, Y1, D0, D1, D_info = data

    subtask = [1, 2, 3] if subtask == "all" else [int(subtask)]

    conv_ns = tf.keras.layers
    norm_ns = tf.keras.layers
    drop_ns = tf.keras.layers
    man_conv_ns = tf.keras.layers

    if task in (2, 3):
        if 1 in subtask:
            conv_ns = layers_keras
        if 2 in subtask:
            norm_ns = layers_keras
        if 3 in subtask:
            drop_ns = layers_keras

    if task == 3:
        man_conv_ns = layers_manual

    args = conv_model.get_default_CNN_model(
        conv_ns=conv_ns, norm_ns=norm_ns, drop_ns=drop_ns, man_conv_ns=man_conv_ns
    )

    if task != 3:
        if epochs is None:
            epochs = args.epochs
        if batch_size is None:
            batch_size = args.batch_size
        X0_sub, Y0_sub = X0, Y0
        X1_sub, Y1_sub = X1, Y1
    else:
        if epochs is None:
            epochs = 2
        if batch_size is None:
            batch_size = 250
        X0_sub, Y0_sub = X0[:250], Y0[:250]
        X1_sub, Y1_sub = X1[:250], Y1[:250]

    # Training model
    print("Starting Model Training")
    history = args.model.fit(
        X0_sub,
        Y0_sub,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X1_sub, Y1_sub),
    )

    return args.model


###############################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--task", default=1, choices="1 2 3 all".split(), help="task to perform"
    )
    parser.add_argument("--subtask", default="all", type=str, help="subtask to perform")
    args = parser.parse_args()

    data = get_data()
    run_task(data, args.task, args.subtask)
