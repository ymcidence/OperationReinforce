from __future__ import division, absolute_import, print_function, unicode_literals

import tensorflow as tf


class InputLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pass
