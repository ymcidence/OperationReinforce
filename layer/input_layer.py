from __future__ import division, absolute_import, print_function, unicode_literals

import tensorflow as tf


class InputLayer(tf.keras.layers.Layer):
    def __init__(self, ns, **kwargs):
        super().__init__(**kwargs)
        self.ns = ns

        # self.cat1_emb = tf.keras.layers.Embedding(self.ns.cat1_num + 1, self.ns.d_model)
        self.uvcc_emb = tf.keras.layers.Embedding(self.ns.uvcc_num + 1, self.ns.d_model)
        self.rank_emb = tf.keras.layers.Embedding(self.ns.max_time, self.ns.d_model)

        self.fc1 = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.ns.d_model)
        ])

    def call(self, inputs, training=True, **kwargs):
        slot_feat = inputs['slot_feat']  # [N T d1]
        meta_feat = inputs['meta_feat'][:, tf.newaxis, :]  # [N 1 d2]
        meta_feat = tf.tile(meta_feat, [1, self.ns.max_time, 1])

        x = tf.concat([slot_feat, meta_feat], axis=-1)

        # cat1 = self.cat1_emb(inputs['cat1'])  # [N T D]
        uvcc = self.uvcc_emb(inputs['uvcc'])[:, tf.newaxis, :]  # [N 1 D]
        uvcc = tf.tile(uvcc, [1, self.ns.max_time, 1])  # [N T D]
        rank = self.rank_emb(inputs['rank'])

        x = self.fc1(x, training=training) + uvcc + rank

        return x
