from __future__ import absolute_import, print_function, division, unicode_literals
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from meta import ROOT_PATH
from util.data.feat import *
from util.revenue import revenue_expected


def _int64_feature(value):
    """Create a feature that is serialized as an int64."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _bytes_feature(value):
    """Create a feature that is stored on disk as a byte array."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _process_slot_feat(file_name, sep='\t', cat1=None):
    df = pd.read_csv(os.path.join(ROOT_PATH, 'data', file_name), sep=sep)

    if cat1 is None:
        cat1 = np.sort(df['effective_category_constraint'].drop_duplicates().values)

    df['cat1'] = _map_dict(df['effective_category_constraint'].values, cat1)

    df['query_cat'] = df['norm_query'] + df['effective_category_constraint'].apply(lambda x: str(x))

    query = df['query_cat'].drop_duplicates().values

    return df, cat1, query


def _map_dict(source, ind, fallback_value=-1):
    rslt = [np.where(ind == s)[0][0] if np.where(ind == s)[0].__len__() > 0 else fallback_value for s in source]
    return rslt


def _trim_or_padding(x, max_time):
    rslt = x
    shape = rslt.shape
    if shape[0] > max_time:
        if shape.__len__() > 1:
            rslt = rslt[:max_time, :]
        else:
            rslt = rslt[:max_time]
        mask = np.ones(max_time)
    elif shape[0] < max_time:
        d = max_time - shape[0]
        padding_shape = d if shape.__len__() <= 1 else [d, shape[1]]
        padding = np.zeros(padding_shape, dtype=rslt.dtype)
        rslt = np.concatenate([rslt, padding], axis=0)

        mask_1 = np.ones(shape[0])
        mask_0 = np.zeros(d)
        mask = np.concatenate([mask_1, mask_0], axis=0)

    else:
        mask = np.ones(max_time)

    return rslt, mask


def _write_line(writer: tf.io.TFRecordWriter, feat_dict: dict):
    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))
    writer.write(example.SerializeToString())


# noinspection PyTypeChecker
def _process(file_name, slot: pd.DataFrame, meta: pd.DataFrame, max_time=6):
    query = meta['query_cat'].drop_duplicates().values

    writer = tf.io.TFRecordWriter(os.path.join(ROOT_PATH, 'data', file_name))

    for i, q in enumerate(query):
        try:
            if i % 100 == 0:
                print(i)
            s: pd.DataFrame = slot[slot['query_cat'] == q].sort_values('rank')  # [T, D]
            m: pd.DataFrame = meta[meta['query_cat'] == q].iloc[0]
            slot_feat = s[SLOT_FEAT_NAME].values  # [T D]
            slot_feat, mask = _trim_or_padding(slot_feat, max_time)
            meta_feat = m[QUERY_FEAT_NAME].values  # [D]
            rank = s['rank'].values  # [T]
            rank, _ = _trim_or_padding(rank, max_time)
            uvcc = m['uvcc']  # scalar
            total_rev, pl_rev, ol_rev = revenue_expected(s)

            features = {
                'slot_feat': _float_feature(slot_feat),
                'meta_feat': _float_feature(meta_feat),
                'rank': _float_feature(rank),
                'uvcc': _int64_feature(uvcc),
                'total_rev': _float_feature(np.asarray([total_rev])),
                'pl_rev': _float_feature(np.asarray([pl_rev])),
                'ol_rev': _float_feature(np.asarray([ol_rev])),
                'mask': _float_feature(mask),
                'query': _bytes_feature(q.encode('utf-8'))
            }
            _write_line(writer, features)
        except IndexError:
            print(IndexError.__name__)
        # except:
        #     print('error not catched')

    writer.close()


def make(task_name, train_slot_file, test_slot_file, query_meta_file, max_time=6):
    train_slot, cat1, train_query = _process_slot_feat(train_slot_file)
    test_slot, _, test_query = _process_slot_feat(test_slot_file, cat1=cat1)
    query_meta = pd.read_csv(os.path.join(ROOT_PATH, 'data', query_meta_file))
    query_meta['query_cat'] = query_meta['Query'] + query_meta['Category'].apply(lambda x: str(x))

    train_meta = query_meta[query_meta['query_cat'].isin(train_query)]
    test_meta = query_meta[query_meta['query_cat'].isin(test_query)]

    uvcc_cat = cat1

    train_meta['uvcc'] = _map_dict(train_meta['Category'], uvcc_cat, uvcc_cat.__len__())
    test_meta['uvcc'] = _map_dict(test_meta['Category'], uvcc_cat, uvcc_cat.__len__())

    meta = {
        'uvcc_dict': uvcc_cat,
        'slot_feat': SLOT_FEAT_NAME,
        'query_feat': QUERY_FEAT_NAME,
        'max_time': max_time
    }

    np.save(os.path.join(ROOT_PATH, 'data', task_name, 'meta.npy'), meta)

    write_name = os.path.join(task_name, 'train.tfrecords')
    _process(write_name, train_slot, train_meta, max_time=max_time)
    write_name = os.path.join(task_name, 'test.tfrecords')
    _process(write_name, test_slot, test_meta, max_time=max_time)


if __name__ == '__main__':
    make('basic1', 'soj_data_train_us_min_impr_50.tsv', 'soj_data_test_us_min_impr_50.tsv',
         'features_query_supply_2020-12-29.csv')
