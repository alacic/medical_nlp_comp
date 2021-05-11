import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import (layers, losses, metrics, models, optimizers,
                              regularizers)
from tensorflow.keras.preprocessing import sequence, text

PADDING_SIZE = 90
LAYER_DTYPE = 'float32'
LABEL_SIZE = 17
EMBEDDING_SIZR = 180
STEPS_PER_EPOCH = 100
EPOCHS = 25
BATCH_SIZE = 64


def get_cnn_model(output_shape=17, name="label", conv_padding="same", cnn_strides=1,
                  cnn_activation="relu", filters=EMBEDDING_SIZR, kernel_size_list=[3, 4, 5, 7, 9]):

    def conv_pool(x, kernel_size):
        x = layers.Conv1D(
            filters,
            kernel_size,
            padding=conv_padding,
            strides=cnn_strides,
            activation=None,
        )(x)
        return x

    def stack_cnn(x):
        cnns = []
        for kernel_size in kernel_size_list:
            cnns.append(conv_pool(x, kernel_size))
        cnn = tf.concat(cnns, axis=-1)
        return cnn
    
    def dila_cnn(x):
        
        x = layers.Conv1D(900, 3, activation=None, dilation_rate=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D()(x)
        x = layers.ReLU()(x)
        # x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(900, 3, activation=None, dilation_rate=2)(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        # x = layers.Dropout(0.1)(x)
        return x

    input_x = layers.Input(shape=(PADDING_SIZE,), dtype=LAYER_DTYPE)

    embedding = layers.Embedding(
        1000,
        EMBEDDING_SIZR,
        mask_zero=True
    )
    x = embedding(input_x)
    mask = embedding.compute_mask(input_x)
    x = layers.GaussianNoise(0.07)(x)  # 0.007
    x = layers.SpatialDropout1D(0.292893218813452)(x)
    x = layers.Dropout(0.292893218813452)(x)
    
    x = layers.Bidirectional(
        layers.GRU(
            PADDING_SIZE,
            return_sequences=True,
        ))(x, mask=mask)

    x = stack_cnn(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = dila_cnn(x)

    output = layers.Dense(
        output_shape,
        activation="sigmoid",
        dtype=LAYER_DTYPE,
        name=name,
        kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3),
    )(x)
    
    model = models.Model(input_x, output)
    return model


def metric_score(y_true, y_pred, eps=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    loss_ = - y_true * tf.math.log(y_pred) - \
        (1 - y_true) * tf.math.log(1 - y_pred)
    loss_ = tf.math.reduce_mean(loss_, axis=-1)
    return 1 - 2 * loss_

def metric_score_type(y_true, y_pred, eps=1e-7):
    mask = (tf.math.reduce_sum(y_true, axis=1) != 0)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    loss_ = - y_true * tf.math.log(y_pred) - \
        (1 - y_true) * tf.math.log(1 - y_pred)
    loss_ = tf.math.reduce_mean(loss_, axis=-1)
    loss_ = tf.boolean_mask(loss_, mask)
    return 1 - 2 * loss_


def loss_region(y_true, y_pred, eps=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    loss_ = - y_true * tf.math.log(y_pred) - \
        (1 - y_true) * tf.math.log(1 - y_pred)
    loss_ = tf.math.reduce_mean(loss_, axis=-1)
    return loss_


def loss_type(y_true, y_pred, eps=1e-7):
    mask = (tf.math.reduce_sum(y_true, axis=1) != 0)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    loss_ = - y_true * tf.math.log(y_pred) - \
        (1 - y_true) * tf.math.log(1 - y_pred)
    loss_ = tf.math.reduce_mean(loss_, axis=-1)
    loss_ = tf.boolean_mask(loss_, mask)
    return loss_


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*50,
    decay_rate=1,
    staircase=False
)

opt = tfa.optimizers.RectifiedAdam(lr_schedule)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=3,
    restore_best_weights=True
)
