from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import nets
import load_data
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('t_model', 'vgg16-like',
                           """Model to use""")
tf.app.flags.DEFINE_string('s_model', 'vgg16-like',
                           """Model to use""")
tf.app.flags.DEFINE_string('data', 'cifar10',
                           """Database to use""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Database to use""")
tf.app.flags.DEFINE_integer('n_epochs', 1000,
                            """Database to use""")

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        data, train_size, test_size, input_shape, nclass = load_data.load(FLAGS.data)
        [X_train, y_train, X_test, y_test] = data

        images = tf.placeholder(tf.float32, input_shape)
        labels = tf.placeholder(tf.int32, [None] + [nclass])

        with tf.variable_scope('Teacher'):
            t_logits = nets.get_modelf(FLAGS.t_model)(images)
            t_loss = nets.teacher_cl_loss(labels, t_logits)

        t_train_op = nets.train_op(t_loss,
                                   utils.get_all_variables_from_scope('Teacher'))

        with tf.variable_scope('Student'):
            s_logits = nets.get_modelf(FLAGS.s_model)(images)
            s_loss = nets.student_cl_loss(t_logits, s_logits)

        s_train_op = nets.train_op(s_loss,
                                   utils.get_all_variables_from_scope('Student'),
                                   global_step)

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.80

        class _LoggerHook(tf.train.SessionRunHook):
          """Logs loss and runtime."""

          def begin(self):
            self._step = -1
            self._start_time = time.time()
            with open(FLAGS.train_dir + '/my_logs.csv') as f:
                f.write('dt,step,train_loss,student_loss\n')

          def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs([t_loss, s_loss])  # Asks for loss value.

          def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_frequency == 0:
              current_time = time.time()
              duration = current_time - self._start_time
              self._start_time = current_time

              t_loss_value = run_values.results[0]
              s_loss_value = run_values.results[1]
            #   print(run_values)
              examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
              sec_per_batch = float(duration / FLAGS.log_frequency)

              format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print (format_str % (datetime.now(), self._step, loss_value,
                                   examples_per_sec, sec_per_batch))

              with open(FLAGS.train_dir + '/my_train_logs.csv', 'a') as f:
                  f.write('{},{},{},{}\n'.format(duration, self._step,
                                               t_loss_value,
                                               s_loss_value))

        with tf.Session() as sess:
            step = 0
            for k in range(FLAGS.n_epochs):
                for X_batch_aug, y_batch in load_data.batch_iterator_train_crop_flip(X_train,
                                                             y_train,
                                                             FLAGS.batch_size):
                    step += 1
                    sess.run([t_train_op, s_train_op], {
                        images: X_batch_aug,
                        labels: y_batch
                    })

                    if (step + 1) % FLAGS.eval_frequency == 0:
                        # s_train_acc = utils.reduce_mean_acc(X_train, y_train,
                        #                                     lambda x: sess.run(s_logits))
                        # t_train_acc = utils.reduce_mean_acc(X_train, y_train,
                        #                                     lambda x: sess.run(t_logits))
                        s_train_acc, t_train_acc = 0, 0
                        s_test_acc = utils.reduce_mean_acc(X_test, y_test,
                                                           lambda x: sess.run(s_logits, {images: x}))
                        t_test_acc = utils.reduce_mean_acc(X_test, y_test,
                                                           lambda x: sess.run(t_logits, {images: x}))


                        with open(FLAGS.train_dir + '/my_eval_logs.csv', 'a') as f:
                            f.write('{},{},{},{},{},{}\n'.format(k, step, t_train_acc,
                                                                 t_test_acc,
                                                                 s_train_acc,
                                                                 s_test_acc))




def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  with open(FLAGS.train_dir + '/my_eval_logs.csv', 'w') as f:
      f.write('epoch,step,t_train_acc,t_test_acc,s_train_acc,s_test_acc\n')

  train()


if __name__ == '__main__':
    tf.app.run()
