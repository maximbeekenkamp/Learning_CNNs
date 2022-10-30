from types import SimpleNamespace

import numpy as np
import tensorflow as tf

###############################################################################################


def get_default_CNN_model(
    conv_ns=tf.keras.layers,
    norm_ns=tf.keras.layers,
    drop_ns=tf.keras.layers,
    man_conv_ns=tf.keras.layers,
):
    """
    Sets up your model architecture and compiles it using the appropriate optimizer, loss, and
    metrics.

    :param conv_ns, norm_ns, drop_ns: what version of this layer to use (either tf.keras.layers or
                                      your implementation from layers_keras)
    :param man_conv_ns: what version of manual Conv2D to use (use tf.keras.layers until you want to
                        test out your manual implementation from layers_manual)

    :returns compiled model
    """

    Conv2D = conv_ns.Conv2D
    BatchNormalization = norm_ns.BatchNormalization
    Dropout = drop_ns.Dropout
    Conv2D_manual = man_conv_ns.Conv2D

    input_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255),
            tf.keras.layers.Resizing(32, 32),
        ]
    )
    output_prep_fn = tf.keras.layers.CategoryEncoding(
        num_tokens=10, output_mode="one_hot"
    )

    ## Augment the input data
    ## https://www.tensorflow.org/guide/keras/preprocessing_layers
    augment_fn = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal")
    ])

    model = CustomSequential(
        [
            Conv2D_manual(16, 3, (2,2), activation="leaky_relu"),
            tf.keras.layers.MaxPool2D(),
            Dropout(0.2),
            Conv2D(64, 3, activation="leaky_relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(), 
            BatchNormalization(), 
            tf.keras.layers.Dense(256, activation="leaky_relu"),
            Dropout(0.2), 
            tf.keras.layers.Dense(64, activation="leaky_relu"), 
            Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        input_prep_fn = input_prep_fn, 
        output_prep_fn = output_prep_fn, 
        augment_fn = augment_fn
    )

    ## Compiles your model using your choice of optimizer
    model.compile(
        optimizer="Adam",  
        loss="categorical_crossentropy", 
        metrics=["categorical_accuracy"],
    )
    return SimpleNamespace(model=model, epochs=20, batch_size=100)


###############################################################################################


class CustomSequential(tf.keras.Sequential):
    """
    Subclasses tf.keras.Sequential to allow us to specify preparation functions that
    will modify input and output data.

    :param input_prep_fn: Modifies input images prior to running the forward pass
    :param output_prep_fn: Modifies input labels prior to running forward pass
    :param augment_fn: Augments input images prior to running forward pass
    """

    def __init__(
        self,
        *args,
        input_prep_fn=lambda x: x,
        output_prep_fn=lambda x: x,
        augment_fn=lambda x: x,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_prep_fn = input_prep_fn
        self.output_prep_fn = output_prep_fn
        self.augment_fn = augment_fn

    def batch_step(self, data, training=False):

        x_raw, y_raw = data

        x = self.input_prep_fn(x_raw)
        y = self.output_prep_fn(y_raw)
        if training:
            x = self.augment_fn(x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=training)
            # Compute the loss value (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if training:
            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return self.batch_step(data, training=True)

    def test_step(self, data):
        return self.batch_step(data, training=False)

    def predict_step(self, inputs):
        x = self.input_prep_fn(inputs)
        return self(x)
