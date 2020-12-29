from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np
import tensorflow as tf


class BinomialSampler(tf.keras.layers.Layer):
    def __init__(self, ns, **kwargs):
        super().__init__(**kwargs)
        self.ns = ns,
        self.temp = ns.temp

    def call(self, inputs, training=True, **kwargs):
        batch_size = tf.shape(inputs)[0]

        # noinspection PyUnresolvedReferences
        eps = tf.random.uniform([batch_size, self.ns.max_time]) if training else .5

        eps = tf.pow(eps, self.temp)

        rslt = tf.greater(tf.stop_gradient(inputs), eps)

        return rslt


class ReplayRewarder(object):
    def __init__(self, meta: dict):
        self.meta = meta

    def __call__(self, inputs: dict, action, **kwargs):
        i_slot_total_rev = self.meta['slot_feat'].index('query_total_rev')
        i_slot_pl_rev = self.meta['slot_feat'].index('query_total_rev_pl_slots')
        i_slot_total_imp = self.meta['slot_feat'].index('query_slot_impr')
        i_slot_pl_imp = self.meta['slot_feat'].index('query_slot_pl_impr')

        slot_feat = inputs['slot_feat'].numpy()

        slot_total_rev = slot_feat[:, :, i_slot_total_rev]
        slot_pl_rev = slot_feat[:, :, i_slot_pl_rev]
        slot_ol_rev = slot_total_rev - slot_pl_rev

        slot_total_imp = slot_feat[:, :, i_slot_total_imp]
        slot_pl_imp = slot_feat[:, :, i_slot_pl_imp]
        slot_ol_imp = slot_total_imp - slot_pl_imp

        replay_pl_rate = slot_total_imp / slot_pl_imp
        replay_pl_rate[np.isinf(replay_pl_rate)] = 1
        replay_pl_rate[np.isnan(replay_pl_rate)] = 1

        replay_ol_rate = slot_total_imp / slot_ol_imp
        replay_ol_rate[np.isinf(replay_ol_rate)] = 1
        replay_ol_rate[np.isnan(replay_ol_rate)] = 1

        replay_pl_rev = slot_pl_rev * replay_pl_rate
        replay_ol_rev = slot_ol_rev * replay_ol_rate

        pl_gate = action.numpy()
        ol_gate = (pl_gate - 1) * -1

        replay_pl_rev = replay_pl_rev * pl_gate
        replay_ol_rev = replay_ol_rev * ol_gate

        replay_rev = np.sum(replay_pl_rev + replay_ol_rev, axis=1)
        replay_pl_rev = np.sum(replay_pl_rev, axis=1)
        replay_ol_rev = np.sum(replay_ol_rev, axis=1)  # [N]

        return replay_rev, replay_pl_rev, replay_ol_rev


class REINFORCE(object):
    def __init__(self, meta: dict, ns):
        self.meta = meta
        self.ns = ns
        self.sampler = BinomialSampler(ns)
        self.rewarder = ReplayRewarder(meta)
        self.sample_time = ns.sample_time
        self.exp_n_pl = ns.exp_n_pl
        self.batch_size = ns.batch_size

    def __call__(self, inputs: dict, logits, training=True):
        """

        :param inputs: a dict from util.data.dataset.Dataset.next()
        :param logits: [N T]
        :return:
        """
        t, p, o, n, c = [], [], [], [], []
        prob = tf.nn.sigmoid(logits)

        st = self.sample_time if training else 2
        for i in range(st):
            action = self.sampler(prob, training=True)

            n_pl = np.sum(action.numpy(), axis=1)
            replay_rev, replay_pl_rev, replay_ol_rev = self.rewarder(inputs, action)

            ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.stop_gradient(action), logits=logits)

            c.append(ce)
            t.append(replay_rev)
            p.append(replay_pl_rev)
            o.append(replay_ol_rev)
            n.append(n_pl)

        t = np.stack(t)  # [T N]
        p = np.stack(p)
        o = np.stack(o)
        c = tf.stack(c)
        n = np.stack(n)  # [T]

        score_function = (t + p) / (n + 1e-8) * self.exp_n_pl
        mean_sf = np.mean(score_function, axis=0)

        score_function = score_function - mean_sf

        loss = tf.reduce_mean(tf.reduce_sum(c * score_function, axis=1), axis=0)

        return loss, np.sum(np.mean(t, axis=0), axis=1), np.sum(np.mean(p, axis=0), axis=1), np.sum(np.mean(o, axis=0),
                                                                                                    axis=1), np.mean(n)

    def update_sampler(self, inc=True):
        if inc:
            self.sampler.temp *= 1.2
        else:
            self.sampler.temp *= 0.8
