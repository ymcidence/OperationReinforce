from __future__ import absolute_import, print_function, division, unicode_literals

import os
import numpy as np
import tensorflow as tf
from meta import ROOT_PATH

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset(object):
    def __init__(self, task_name, batch_size, training=True):
        self.task_name = task_name
        self.batch_size = batch_size
        self.training = training
        self.meta = np.load(os.path.join(ROOT_PATH, 'data', self.task_name, 'meta.npy'), allow_pickle=True).item()
        # noinspection PyUnresolvedReferences
        self.max_time = self.meta['max_time']
        # noinspection PyUnresolvedReferences
        self.uvcc_num = self.meta['uvcc_dict'].__len__()
        self._load_data()

    def _load_data(self):
        # noinspection PyUnresolvedReferences
        slot_feat = self.meta['slot_feat']
        # noinspection PyUnresolvedReferences
        query_feat = self.meta['query_feat']

        def _map(e: tf.train.Example):
            feat_dict = {
                'slot_feat': tf.io.FixedLenFeature([self.max_time * slot_feat.__len__()], tf.float32),
                'meta_feat': tf.io.FixedLenFeature([query_feat.__len__()], tf.float32),
                'rank': tf.io.FixedLenFeature([self.max_time], tf.float32),
                # 'cat1': tf.io.FixedLenFeature([self.max_time], tf.float32),
                'uvcc': tf.io.FixedLenFeature([], tf.int64),
                'total_rev': tf.io.FixedLenFeature([1], tf.float32),
                'pl_rev': tf.io.FixedLenFeature([1], tf.float32),
                'ol_rev': tf.io.FixedLenFeature([1], tf.float32),
                'mask': tf.io.FixedLenFeature([self.max_time], tf.float32),
                'query': tf.io.FixedLenFeature([], tf.string)}

            features = tf.io.parse_single_example(e, features=feat_dict)

            features['slot_feat'] = tf.reshape(features['slot_feat'], [self.max_time, -1])
            features['rank'] = tf.cast(features['rank'], tf.int32)
            # features['cat1'] = tf.cast(features['cat1'], tf.int32)
            features['uvcc'] = tf.cast(features['uvcc'], tf.int32)
            features['mask'] = tf.cast(features['mask'], tf.int32)

            return features

        def _read_record(file_name):
            return tf.data.TFRecordDataset(file_name) \
                .map(_map, num_parallel_calls=AUTOTUNE) \
                .prefetch(AUTOTUNE)

        train_record = os.path.join(ROOT_PATH, 'data', self.task_name, 'train.tfrecords')
        test_record = os.path.join(ROOT_PATH, 'data', self.task_name, 'test.tfrecords')

        train_data = _read_record(train_record).cache().shuffle(10000).batch(self.batch_size)
        test_data = _read_record(test_record).cache()

        if self.training:
            test_data = test_data.shuffle(10000).batch(self.batch_size)
        else:
            test_data = test_data.batch(1)

        self.train_data = train_data
        self.test_data = test_data
