from __future__ import absolute_import, print_function, division, unicode_literals

from abc import ABC

import tensorflow as tf
from layer.encodec import layer_function
from layer.input_layer import InputLayer
from layer.attention import create_padding_mask


class BasicModel(tf.keras.Model, ABC):
    def __init__(self, ns, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ns = ns

        self.input_layer = InputLayer(ns)
        self.enc_layer = [
            layer_function(True, ns.d_model, ns.n_head, ns.d_model * 2, ns.rate, att='rga', max_seq=ns.max_time) for _
            in range(ns.n_layer)]
        self.fc = tf.keras.Sequential([tf.keras.layers.Dropout(ns.rate),
                                       tf.keras.layers.Dense(1)])

    def call(self, inputs, training=True, mask=None):
        x = self.input_layer(inputs, training=training)  # [N T D]
        mask = create_padding_mask(inputs['rank'], tar=0) if mask is None else mask

        for layer in self.enc_layer:
            x, _ = layer(x, training=training, mask=mask)

        return tf.squeeze(self.fc(x, training=training), axis=-1)
