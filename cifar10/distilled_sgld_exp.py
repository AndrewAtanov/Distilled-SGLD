from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import sys
import numpy as np
import tensorflow as tf

import my_cifar10 as cifar10
from nets import vgg16

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_distilled_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_bool('s_dropout', False,
                         """How often to log results to the console.""")


def evaluate_once():


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    t_images, labels = cifar10.distorted_inputs()
    is_training = tf.placeholder(tf.bool)
    with tf.variable_scope('Teacher'):
        t_global_step = tf.contrib.framework.get_or_create_global_step()
        t_logits = vgg16(t_images, is_training=is_training)
        tf.summary.histogram('Teacher softmax', tf.nn.softmax(t_logits/FLAGS.T))
        t_loss = cifar10.loss(t_logits, labels)
        tf.summary.scalar('Teacher loss (on batch)', t_loss)
        tf.summary.scalar('Accuracy',
                          tf.reduce_mean(tf.to_float(tf.argmax(labels, axis=1) == tf.argmax(t_logits, axis=1))))


    t_train_op = cifar10.train(t_loss, t_global_step)

    with tf.variable_scope('Student'):
        s_global_step = tf.contrib.framework.get_or_create_global_step()
        # s_images, _ = cifar10.distorted_inputs()
        s_logits = vgg16(t_images, use_dropout=FLAGS.s_dropout,
                         is_training=is_training)
        s_loss = cifar10.student_loss(t_logits, s_logits)
        tf.summary.scalar('Student loss (on batch)', s_loss)
        tf.summary.scalar('Accuracy',
                          tf.reduce_mean(tf.to_float(tf.argmax(labels, axis=1) == tf.argmax(s_logits, axis=1))))

    s_train_op = cifar10.train(s_loss, s_global_step)

    # EVAL section

    t_labels = tf.to_int32(tf.argmax(t_logits, axis=1))
    s_labels = tf.to_int32(tf.argmax(s_logits, axis=1))

    mask = tf.logical_or(tf.not_equal(labels, s_labels),
                         tf.not_equal(labels, t_labels))

    summ_imgs = tf.boolean_mask(t_images, mask)
    summ_labels = tf.boolean_mask(labels, mask)

    t_top_k_op = tf.nn.in_top_k(t_logits, labels, 1)
    s_top_k_op = tf.nn.in_top_k(s_logits, labels, 1)

    s_summary_writer = tf.summary.FileWriter(FLAGS.train_dir + '/student')
    t_summary_writer = tf.summary.FileWriter(FLAGS.train_dir + '/teacher')


    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

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

          format_str = ('%s: step %d, teacher loss = %.2f, student loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step,
                               t_loss_value, s_loss_value,
                               examples_per_sec, sec_per_batch))


    session_config = tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement,
        allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8


    # summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

    summary_op = tf.summary.merge_all()

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(t_loss),
               _LoggerHook(),
               tf.train.SummarySaverHook(save_steps=100,
                                         output_dir=FLAGS.train_dir,
                                         summary_op=summary_op)],
        config=session_config) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run([t_train_op, s_train_op])


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  with open(FLAGS.train_dir + '/desc', 'w') as f:
      f.write(str(FLAGS.__flags))
  train()


if __name__ == '__main__':
  tf.app.run()
