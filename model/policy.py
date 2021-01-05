from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np
import tensorflow as tf


class MMD(object):
    def __init__(self, z_dim, kernel='IMQ'):
        self.kernel = kernel
        self.z_dim = z_dim

    def mmd_penalty(self, sample_qz, sample_pz):
        # opts = self.opts
        kernel = self.kernel
        n = tf.shape(sample_qz)[0]
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

            res1 = tf.exp(- distances_qz / 2. / sigma2_k)
            res1 += tf.exp(- distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp(- distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

            c_base = self.z_dim
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                cc = c_base * scale
                res1 = cc / (cc + distances_qz)
                res1 += cc / (cc + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = cc / (cc + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        else:
            raise Exception('the kernel is not implemented')
        return stat

    def __call__(self, feat: tf.Tensor, temp):
        eps = tf.random.uniform(tf.shape(feat), dtype=tf.float32)
        eps = tf.pow(eps, temp)
        eps = tf.cast(tf.greater(eps, .5), tf.float32)

        return tf.reduce_sum(self.mmd_penalty(feat, eps))


class BinomialSampler(tf.keras.layers.Layer):
    def __init__(self, ns, **kwargs):
        super().__init__(**kwargs)
        self.ns = ns
        self.temp = ns.temp

    def call(self, inputs, training=True, **kwargs):
        batch_size = tf.shape(inputs)[0]

        # noinspection PyUnresolvedReferences
        eps = tf.random.uniform([batch_size, self.ns.max_time], dtype=tf.float32) if training else .5

        eps = tf.pow(eps, self.temp)

        rslt = tf.greater(tf.stop_gradient(inputs), eps)
        rslt = tf.cast(rslt, tf.float32)

        return rslt


class ReplayRewarder(object):
    def __init__(self, meta: dict):
        self.meta = meta

    def __call__(self, inputs: dict, action, **kwargs):
        i_slot_total_rev = self.meta['slot_feat'].index('query_slot_total_rev')
        i_slot_pl_rev = self.meta['slot_feat'].index('query_slot_total_rev_pl_slots')
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
        self.mmd = MMD(z_dim=ns.max_time)
        self.sample_time = ns.sample_time
        self.exp_n_pl = ns.exp_n_pl
        self.batch_size = ns.batch_size

    def __call__(self, inputs: dict, logits, training=True):
        """

        :param inputs: a dict from util.data.dataset.Dataset.next()
        :param logits: [N L]
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
        c = tf.reduce_sum(tf.stack(c), axis=-1)
        n = np.stack(n)  # [T]

        score_function = (t + p) / (n + 1e-8) * self.exp_n_pl
        mean_sf = np.mean(score_function, axis=0)

        score_function = score_function - mean_sf

        mmd_loss = self.mmd(prob, self.sampler.temp)

        loss = tf.reduce_mean(c * score_function) + mmd_loss

        return loss, mmd_loss, np.sum(np.mean(t, axis=0)), np.sum(np.mean(p, axis=0)), np.sum(
            np.mean(o, axis=0)), np.mean(n)

    def update_sampler(self, inc=True):
        if inc:
            self.sampler.temp *= 1.2
        else:
            self.sampler.temp *= 0.8
