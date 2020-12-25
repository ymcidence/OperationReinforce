from __future__ import absolute_import, print_function, division, unicode_literals

from abc import ABC

import tensorflow as tf
from layer.encodec import layer_function
from layer.input_layer import InputLayer


class BasicModel(tf.keras.Model, ABC):
    def __init__(self, n_layer, d_model, n_head, rate=.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_layer = InputLayer()
        self.enc_layer = [layer_function(True, d_model, n_head, d_model * 2, rate, att='rga', max_seq=6) for _ in
                          range(n_layer)]
        self.fc = tf.keras.Sequential([tf.keras.layers.Dropout(rate),
                                       tf.keras.layers.Dense(1)])

    def call(self, inputs, training=True, mask=None):
        x = self.input_layer(inputs, training=training)  # [N T D]
