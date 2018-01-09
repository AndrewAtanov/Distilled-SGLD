from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import nets
import load_data
import utils
import os
import time
from utils import timing
from LangevinSampler import RMSPropLangevin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('t_model', 'vgg16-like',
                           """Model to use""")
tf.app.flags.DEFINE_string('s_model', 'vgg16-like',
                           """Model to use""")
tf.app.flags.DEFINE_string('lr_strategy', 'linear',
                           """ """)
tf.app.flags.DEFINE_string('data', 'cifar10',
                           """Database to use""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                           """Database to use""")
tf.app.flags.DEFINE_integer('eval_frequency', 100,
                           """Database to use""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Database to use""")
tf.app.flags.DEFINE_integer('n_epochs', 200,
                            """Database to use""")
tf.app.flags.DEFINE_integer('ensemble_step', 50,
                            """ """)
tf.app.flags.DEFINE_integer('s_step', 1,
                            """ SNN train freaquency """)
tf.app.flags.DEFINE_integer('start_snn', 0,
                            """ SNN train start epoch """)
tf.app.flags.DEFINE_integer('start_snn2', 0,
                            """ SNN train start epoch """)
tf.app.flags.DEFINE_integer('save_model_frequency', 500,
                            """ """)
tf.app.flags.DEFINE_integer('lr_to0_from', 100,
                            """ """)
tf.app.flags.DEFINE_float('lr', 0.001,
                          """learning rate""")
tf.app.flags.DEFINE_float('decay_step', 1000,
                          """decay step for exponential strategy""")
tf.app.flags.DEFINE_float('decay_rate', 0.9,
                          """decay rate for exponential strategy""")
tf.app.flags.DEFINE_bool('sgld_target', False,
                         """ Train Ensemble-Student on ensemble target """)
tf.app.flags.DEFINE_bool('go_deeper', False,
                         """ Train SNN2 on SNN ensemble """)
tf.app.flags.DEFINE_bool('make_sgld_target', False,
                         """ Train Ensemble-Student on ensemble target """)
tf.app.flags.DEFINE_float('t_vggsize', 1.,
                          """ TNN VGG size """)
tf.app.flags.DEFINE_float('s_vggsize', 1.,
                          """ SNN VGG size """)
tf.app.flags.DEFINE_float('reg', 1.,
                          """ L2 loss hyperparameter """)

class PseudoEnsemble:
    def __init__(self, y):
        self.n_estimators = 0
        self._y = y.copy()

    def add_estimator(self, y):
        self._y = self._y * (self.n_estimators * 1. / (self.n_estimators + 1))
        self.n_estimators += 1
        self._y += y / (self.n_estimators * 1.)

    def accuracy(self, true_y):
        return np.mean(true_y == self._y.argmax(axis=1))


@timing
def get_logist(data, nclasses, sess, is_training, logits, inputs,
               bs=1000):
    ans = np.zeros((len(data), nclasses))
    for i in range(0, len(data), bs):
        ans[i: i+bs] = sess.run(logits, {
            is_training: False,
            inputs: data[i: i+bs]
        })
    return ans



def get_eval_once(data, y, imgs_pure,
                  sess, is_training, inputs,
                  t_logits, s_logits, es_logits, ds_logits,
                  ensemble, mode='test'):
    def foo(epoch, step):
        s_true_count, t_true_count = 0, 0
        k = 0
        # print(data)
        for x_batch, y_batch, img_batch in load_data.iterate_minibatches_trio(data, y,
                                                                              imgs_pure,
                                                                              batchsize=1000):
            k += 1
            if FLAGS.go_deeper:
                t_logits_val, s_logits_val, ds_logits_val = sess.run([t_logits, s_logits, ds_logits], {
                    is_training: False,
                    inputs: x_batch
                })
            else:
                t_logits_val, s_logits_val = sess.run([t_logits, s_logits], {
                    is_training: False,
                    inputs: x_batch
                })


            gt_labels = y_batch.argmax(axis=1)
            s_labels = s_logits_val.argmax(axis=1)
            t_labels = t_logits_val.argmax(axis=1)

            if FLAGS.go_deeper:
                ds_labels = ds_logits_val.argmax(axis=1)
                ds_true_count = np.sum(gt_labels == ds_labels)

            s_true_count += np.sum(gt_labels == s_labels)
            t_true_count += np.sum(gt_labels == t_labels)

            if FLAGS.go_deeper:
                ds_true_count += np.sum(gt_labels == ds_labels)

            mask = np.logical_or(s_labels != gt_labels, t_labels != gt_labels)
            if mask.any():
                np.save(FLAGS.train_dir + '/{}-miss-imgs{}.npy'.format(mode, k),
                        img_batch[mask])
                np.save(FLAGS.train_dir + '/{}-miss-labels{}.npy'.format(mode, k),
                        gt_labels[mask])
                np.save(FLAGS.train_dir + '/{}-miss-t_logits{}.npy'.format(mode, k),
                        t_logits_val[mask])
                np.save(FLAGS.train_dir + '/{}-miss-s_logits{}.npy'.format(mode,k),
                        s_logits_val[mask])

        s_test_acc = s_true_count * 1. / len(y)
        t_test_acc = t_true_count * 1. / len(y)

        ds_test_acc = 0
        if FLAGS.go_deeper:
            ds_test_acc = ds_true_count * 1. / len(y)

        with open(FLAGS.train_dir + '/my_{}_eval_logs.csv'.format(mode),
                  'a') as f:
            f.write('{},{},{},{},{},{},{},{}\n'.format(epoch, step, 0,
                                                       t_test_acc,
                                                       0,
                                                       s_test_acc,
                                                       ensemble.accuracy(y.argmax(axis=1)),
                                                       ds_test_acc))

        return t_test_acc

    return foo


def train():
    with tf.Graph().as_default():
        # global_step = tf.contrib.framework.get_or_create_global_step()

        data, train_size, test_size, input_shape, nclass = load_data.load(FLAGS.data)
        [X_train, y_train, X_test, y_test] = data

        imgs_tr, _, imgs_te, _ = load_data.load_pure(FLAGS.data)


        global_step = tf.Variable(0, trainable=False)
        learning_rate_placeholder = tf.placeholder(tf.float32)
        if FLAGS.lr_strategy == 'exp':
            learning_rate = tf.train.exponential_decay(learning_rate_placeholder,
                                                       global_step,
                                                       FLAGS.decay_step,
                                                       FLAGS.decay_rate,
                                                       staircase=True)
        else:
            learning_rate = learning_rate_placeholder
        images = tf.placeholder(tf.float32, input_shape)
        labels = tf.placeholder(tf.int32, [None] + [nclass])
        is_training = tf.placeholder(tf.bool)

        with tf.variable_scope('Teacher'):
            t_logits = nets.get_modelf(FLAGS.t_model)(images,
                                                      is_training=is_training,
                                                      nclasses=nclass,
                                                      k=FLAGS.t_vggsize)
            t_loss = nets.teacher_cl_loss(labels, t_logits)

        t_train_op = nets.t_train_op(t_loss,
                                     utils.get_all_variables_from_scope('Teacher'),
                                     learning_rate=learning_rate_placeholder,
                                     global_step=global_step)

        if FLAGS.s_model != 'no':
            with tf.variable_scope('Student'):
                s_logits = nets.get_modelf(FLAGS.s_model)(images,
                                                          is_training=is_training,
                                                          nclasses=nclass,
                                                          k=FLAGS.s_vggsize)
                s_loss = nets.student_cl_loss(t_logits, s_logits)

            s_train_op = nets.train_op(s_loss,
                                       utils.get_all_variables_from_scope('Student'),
                                       learning_rate=learning_rate_placeholder)
        else:
            s_train_op = tf.no_op()
            s_loss = t_loss
            s_logits = t_logits

        ds_logits = None

        if FLAGS.go_deeper:
            with tf.variable_scope('Deep-Student'):
                ds_logits = nets.get_modelf(FLAGS.s_model)(images,
                                                           is_training=is_training,
                                                           nclasses=nclass,
                                                           k=FLAGS.s_vggsize)
                ds_loss = nets.student_cl_loss(s_logits, ds_logits)

            ds_train_op = nets.train_op(ds_loss,
                                        utils.get_all_variables_from_scope('Deep-Student'),
                                        learning_rate=learning_rate_placeholder)



        es_logits = None

        # if FLAGS.sgld_target:
        #     ensemble_logits = tf.placeholder(tf.float32, [None] + [nclass])
        #     with tf.variable_scope('Ensemble-Student'):
        #         es_logits = nets.get_modelf(FLAGS.s_model)(images,
        #                                                    is_training=is_training,
        #                                                    nclasses=nclass)
        #         es_loss = nets.student_cl_loss(ensemble_logits, es_logits)
        #     es_train_op = nets.train_op(es_loss,
        #                                 utils.get_all_variables_from_scope('Ensemble-Student'),
        #                                 learning_rate=learning_rate)




        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.80

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            n_iter_to_train = FLAGS.n_epochs * train_size / FLAGS.batch_size
            n_iter_per_epoch = train_size / FLAGS.batch_size
            ensemble = PseudoEnsemble(np.zeros_like(y_test))
            train_ensemble = PseudoEnsemble(np.zeros_like(y_train))


            eval_once = get_eval_once(X_test, y_test, imgs_te,
                                      sess, is_training, images,
                                      t_logits, s_logits, es_logits, ds_logits,
                                      ensemble)

            tr_eval_once = get_eval_once(X_train, y_train, imgs_tr,
                                      sess, is_training, images,
                                      t_logits, s_logits, es_logits, ds_logits,
                                      ensemble, mode='train')
            step = 0

            sess.run(init_op)

            es_loss_val = -1
            k = 0

            for k in range(FLAGS.n_epochs):
                for X_batch_aug, y_batch, idxs in load_data.batch_iterator_train_crop_flip(X_train,
                                                                                           y_train,
                                                                                           FLAGS.batch_size):
                    step += 1
                    _lr = FLAGS.lr
                    if FLAGS.lr_strategy == 'linear':
                        _lr = FLAGS.lr if k < FLAGS.lr_to0_from else FLAGS.lr * (n_iter_to_train - step) / (n_iter_to_train - FLAGS.lr_to0_from * n_iter_per_epoch)
                        if k > 190:
                            _lr = 0.00001

                    if FLAGS.lr_strategy == 'step':
                        if k < 100:
                            _lr = 0.001
                        elif k < 200:
                            _lr = 0.0001
                        elif k < 300:
                            _lr = 1e-5
                        else:
                            _lr = 1e-6

                    if _lr < 0:
                        break

                    # print('learning rate', _lr)

                    t_loss_val, s_loss_val, _, _ = sess.run([t_loss, s_loss, t_train_op, s_train_op], {
                        images: X_batch_aug,
                        labels: y_batch,
                        is_training: True,
                        learning_rate_placeholder: _lr
                    })

                    # if step % FLAGS.s_step == 0 and k >= FLAGS.start_snn:
                    #     s_loss_val, _ = sess.run([s_loss, s_train_op], {
                    #         images: X_batch_aug,
                    #         labels: y_batch,
                    #         is_training: True,
                    #         learning_rate: _lr
                    #     })

                    if FLAGS.go_deeper and step % FLAGS.s_step == 0 and k >= FLAGS.start_snn2:
                        ds_loss_val, _ = sess.run([ds_loss, ds_train_op], {
                            images: X_batch_aug,
                            labels: y_batch,
                            is_training: True,
                            learning_rate_placeholder: _lr
                        })

                    # if FLAGS.sgld_target:
                    #     es_loss_val, _ = sess.run([es_loss, es_train_op], {
                    #         images: X_batch_aug,
                    #         ensemble_logits: train_ensemble._y[idxs],
                    #         is_training: True,
                    #         learning_rate: _lr
                    #     })

                    with open(FLAGS.train_dir + '/my_train_logs.csv', 'a') as f:
                        f.write('{},{},{},{},{}\n'.format(k, step, t_loss_val,
                                                          s_loss_val, es_loss_val))

                    if FLAGS.ensemble_step != -1 and step % FLAGS.ensemble_step == 0:
                        t_logits_val = get_logist(X_test, nclass, sess,
                                                  is_training, t_logits, images)
                        ensemble.add_estimator(t_logits_val)

                        if FLAGS.sgld_target or FLAGS.make_sgld_target:
                            t_logits_val = get_logist(X_train, nclass, sess,
                                                      is_training, t_logits, images)
                            train_ensemble.add_estimator(t_logits_val)
                            np.save(FLAGS.train_dir + 'ensemble-train-logits.npy',
                                    train_ensemble._y)

                    if step % FLAGS.eval_frequency == 0:
                        t_acc = eval_once(k, step)

                    if k % FLAGS.save_model_frequency == 0:
                        saver.save(sess, FLAGS.train_dir + '/model.ckpt')

            # tr_eval_once(k, step)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)

    tf.gfile.MakeDirs(FLAGS.train_dir)

    with open(FLAGS.train_dir + '/my_test_eval_logs.csv', 'w') as f:
        f.write('epoch,step,t_train_acc,t_test_acc,s_train_acc,s_test_acc,ensemble_test_acc,Deep-SNN_test_acc\n')

    with open(FLAGS.train_dir + '/my_train_eval_logs.csv', 'w') as f:
        f.write('epoch,step,t_train_acc,t_test_acc,s_train_acc,s_test_acc,ensemble_test_acc,Deep-SNN_test_acc\n')

    with open(FLAGS.train_dir + '/my_train_logs.csv', 'w') as f:
        f.write('epoch,step,train_loss,student_loss,es_loss\n')

    with open(FLAGS.train_dir + '/desc', 'w') as f:
        f.write(str(FLAGS.__flags))

    train()

if __name__ == '__main__':
    tf.app.run()
