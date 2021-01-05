from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from model.basic_model import BasicModel as Model
from model.policy import REINFORCE
from util.config import parser
from util.data.dataset import Dataset
from util.train_helper import prepare_training


def hook(model: Model, policy: REINFORCE, dataset: Dataset, step):
    l, t, p, o, n = 0, 0, 0, 0, 0
    gt, gp = 0, 0

    count = 0
    for x in dataset.test_data:
        logits = model(x, training=False)
        loss, mmd, _t, _p, _o, _n = policy(x, logits)
        l += loss
        t += _t
        p += _p
        o += _o
        n += _n

        gt += tf.reduce_sum(x['total_rev']).numpy()
        gp += tf.reduce_sum(x['pl_rev'])
        count += 1

    l = l / count
    tf.summary.scalar('test/loss', l, step=step)
    tf.summary.scalar('test/total_rev', t, step=step)
    tf.summary.scalar('test/pl_rev', p, step=step)
    tf.summary.scalar('test/ol_rev', o, step=step)
    tf.summary.scalar('test/n', n, step=step)
    tf.summary.scalar('test/g_total_rev', gt, step=step)
    tf.summary.scalar('test/g_pl_rev', gp, step=step)
    tf.summary.scalar('test/temp', policy.sampler.temp, step=step)

    return n


def main():
    ns = parser.parse_args()
    result_path, save_path, summary_path = prepare_training(ns.task_name)
    dataset = Dataset(ns.task_name, ns.batch_size)
    ns.uvcc_num = dataset.uvcc_num
    model = Model(ns)
    policy = REINFORCE(dataset.meta, ns)
    opt = tf.keras.optimizers.Adam(ns.learning_rate)
    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(opt=opt, model=model)
    step = 0
    max_n = 900000
    for i in range(ns.epoch):
        with writer.as_default():
            for x in dataset.train_data:
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss, mmd, t, p, o, n = policy(x, logits)
                    g = tape.gradient(loss, sources=model.trainable_variables)
                    opt.apply_gradients(zip(g, model.trainable_variables))

                if step % 100 == 0:
                    print('epoch {} step {}: loss {} t {} p {} o {} n {}'.format(i, step, loss, t, p, o, n))
                    tf.summary.scalar('train/batch_revenue', t, step=step)
                    tf.summary.scalar('train/batch_pl_rev', p, step=step)
                    tf.summary.scalar('train/batch_ol_rev', o, step=step)
                    tf.summary.scalar('train/batch_n_pl', n, step=step)
                    tf.summary.scalar('train/loss', loss, step=step)
                    tf.summary.scalar('train/mmd', mmd, step=step)

                step += 1
            test_n = hook(model, policy, dataset, step)

            if max_n < test_n:
                policy.update_sampler()

            elif max_n >= 1.1 * test_n:
                max_n = 1.1 * test_n

            save_name = os.path.join(save_path, 'ym' + str(i))
            checkpoint.save(file_prefix=save_name)


if __name__ == '__main__':
    main()
