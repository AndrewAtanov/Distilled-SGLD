# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import my_cifar10 as cifar10
from nets import vgg16

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('model_eval', 'student',
                           """teacher or student""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_bool('s_dropout', False,
                         """How often to log results to the console.""")
tf.app.flags.DEFINE_bool('read_params', False,
                         """How often to log results to the console.""")

LABEL_TO_STR = "airplane automobile bird cat deer dog frog horse ship truck".split()


def eval_once(saver, t_summary_writer, s_summary_writer,
              t_top_k_op, s_top_k_op,
              summary_op, **kwargs):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  session_config = tf.ConfigProto(allow_soft_placement=True)
  session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
  with tf.Session(config=session_config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      print('NUM ITER = {}'.format(num_iter))
      t_true_count = 0  # Counts the number of correct predictions.
      s_true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        summ_imgs, summ_labels, t_logits_val, s_logits_val, predictions = sess.run([kwargs['summ_imgs'],
                                                        kwargs['summ_labels'],
                                                        kwargs['t_logits'],
                                                        kwargs['s_logits'],
                                                        [t_top_k_op, s_top_k_op]])
        t_true_count += np.sum(predictions[0])
        s_true_count += np.sum(predictions[1])

        np.save(FLAGS.eval_dir + '/miss-imgs{}.npy'.format(step),
                summ_imgs)
        np.save(FLAGS.eval_dir + '/miss-labels{}.npy'.format(step),
                summ_labels)
        np.save(FLAGS.eval_dir + '/miss-t_logits{}.npy'.format(step),
                t_logits_val)
        np.save(FLAGS.eval_dir + '/miss-s_logits{}.npy'.format(step),
                s_logits_val)

        step += 1

      # Compute precision @ 1.
      t_precision = t_true_count / total_sample_count
      s_precision = s_true_count / total_sample_count
      print('%s: teacher precision @ 1 = %.3f' % (datetime.now(), t_precision))
      print('%s: student precision @ 1 = %.3f' % (datetime.now(), s_precision))

      summary = tf.Summary()
    #   summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=t_precision)
      t_summary_writer.add_summary(summary, global_step)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=s_precision)
      s_summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'

    t_images, labels = cifar10.inputs(eval_data=eval_data)
    with tf.variable_scope('Teacher'):
        t_logits = vgg16(t_images, is_training=False)

    with tf.variable_scope('Student'):
        # s_images, _ = cifar10.distorted_inputs()
        s_logits = vgg16(t_images, is_training=False, use_dropout=FLAGS.s_dropout)

    label_to_name_tensor = tf.constant(LABEL_TO_STR)

    t_labels = tf.to_int32(tf.argmax(t_logits, axis=1))
    s_labels = tf.to_int32(tf.argmax(s_logits, axis=1))

    mask = tf.logical_or(tf.not_equal(labels, s_labels),
                         tf.not_equal(labels, t_labels))

    summ_imgs = tf.boolean_mask(t_images, mask)
    summ_labels = tf.boolean_mask(labels, mask)

    # tf.map_fn(lambda idx: tf.summary.image('', summ_imgs[idx:idx+1]),
    #           tf.range(0, tf.shape(summ_imgs)[0]))
    # tf.summary.image('TNN -, SNN+', summ_imgs, max_outputs=10)



    # images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.

    # if FLAGS.model_eval == 'teacher':
    #     logits = t_logits
    # elif FLAGS.model_eval == 'student':
    #     logits = s_logits
    # else:
    #     raise NotImplementedError('No such model!')

    # Calculate predictions.
    t_top_k_op = tf.nn.in_top_k(t_logits, labels, 1)
    s_top_k_op = tf.nn.in_top_k(s_logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    s_summary_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/student', g)
    t_summary_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/teacher', g)

    while True:
      eval_once(saver, t_summary_writer, s_summary_writer,
                t_top_k_op, s_top_k_op, summary_op,
                summ_imgs=summ_imgs, summ_labels=summ_labels,
                t_logits=tf.boolean_mask(t_logits, mask),
                s_logits=tf.boolean_mask(s_logits, mask))
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  with open(FLAGS.checkpoint_dir + '/desc', 'r') as f:
      d = eval(f.readline())
      FLAGS.s_dropout = d['s_dropout']
      print('Student Dropout {}'.format(FLAGS.s_dropout))

  if FLAGS.eval_dir == '':
    FLAGS.eval_dir = FLAGS.checkpoint_dir.rstrip('/') + '-' + FLAGS.eval_data + '/'

  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
