import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from mnist_nets import mnist_lenet5
from LangevinSampler import LangevinSampler, RMSPropLangevin
from utils import *
import os
from itertools import count
import argparse

LOG_START = 'iteration,cross_entropy,train_accuracy,test_accuracy,sgld_test_accuracy\n'
LOG_DESCRIPTOR = '{},'*4 + '{}\n'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', default='~/atanov/data/MNIST_data/')
parser.add_argument('--summ_dir', default=None)
parser.add_argument('--keep_prob', default=0.5, type=float)
parser.add_argument('--epoch', default=int(1000), type=int)
parser.add_argument('--log', default=int(100), type=int)
parser.add_argument('--burn', default=int(1000), type=int)
parser.add_argument('--tbs', default=int(10), type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--noise_ratio', default=2, type=float)
args = parser.parse_args()

log_file_name = None

EXP_INFO = '# "keep_prob" = {keep_prob}, "batch_size" = {tbs}, "lr" = {lr}, "noise_ratio": {noise_ratio}, "burn": {burn}\n'

if args.summ_dir is None:
    for i in count():
        name = 'mnist-LeNet-SGLD-{}'.format(i)
        if not os.path.exists(name):
            # os.makedirs('./{}'.format(i))
            # args.summ_dir = './{}'.format(i)
            log_file_name = name
            with open(name, 'w') as f:
                f.write(EXP_INFO.format(**vars(args)))
                f.write(LOG_START)
            break

mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
x_train = np.concatenate((mnist.train.images, mnist.validation.images), axis=0).reshape((-1, 28, 28, 1))
y_train = np.concatenate((mnist.train.labels, mnist.validation.labels), axis=0)

x_test = mnist.test.images.copy().reshape((-1, 28, 28, 1))
y_test = mnist.test.labels.copy()

# train_mdoe = tf.placeholder_with_default(True, [])
mode = tf.placeholder(tf.string, [])
input_x = tf.placeholder(tf.float32, [None, 28, 28, 1])
input_y = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope('Teacher'):
    logits = mnist_lenet5(input_x, 'Teacher', tf.nn.relu,
                          args.keep_prob, tf.equal(mode, 'Train'))
    proba = tf.nn.softmax(logits)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=input_y,
                                                logits=logits))

print(logits.dtype, input_y.dtype)

n_models = tf.get_variable('n_models', shape=[], initializer=tf.constant_initializer(0, dtype=tf.int32),
                           dtype=tf.int32)
test_avg_pred = tf.get_variable('test_avg_pred', shape=y_test.shape,
                                initializer=tf.zeros_initializer())

with tf.control_dependencies([tf.assign_add(n_models, tf.to_int32(1))]):
    upd_test_pred = tf.assign(test_avg_pred,
                              (test_avg_pred * tf.to_float(n_models - 1) + proba) / tf.to_float(n_models))

sgld_test_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(test_avg_pred, 1),
                                                         tf.argmax(input_y, 1))))

teacher_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, 1),
                                                       tf.argmax(input_y, 1))))

optimizer = RMSPropLangevin(args.lr, tf.to_double(args.noise_ratio))
teacher_train_op, langevin_step = optimizer.sample(cross_entropy,
                                                   var_list=get_all_variables_from_scope('Teacher'))
#
# optimizer = tf.train.RMSPropOptimizer(args.lr)
# teacher_train_op = optimizer.minimize(cross_entropy,
#                                       var_list=get_all_variables_from_scope('Teacher'))

cross_entropy_summary = tf.summary.scalar('cross_entropy', cross_entropy)
train_acc_summ = tf.summary.scalar('Train', teacher_accuracy)
test_acc_summ = tf.summary.scalar('Test', teacher_accuracy)



session = tf.Session()
session.run(tf.global_variables_initializer())


teacher_batch_size = args.tbs
start_langevin = False

for k, [batch_xs, batch_ys] in enumerate(batch_iterator(x_train, y_train,
                                                        args.tbs,
                                                        epochs=args.epoch)):
    session.run(langevin_step if start_langevin else teacher_train_op, {
        input_x: batch_xs,
        input_y: batch_ys,
        mode: 'Train',
    })

    if (k + 1) % args.burn == 0:
        session.run(upd_test_pred, {
            mode: 'Test',
            input_x: x_test,
        })

    if (k + 1) % args.log == 0:
        test_acc, sgld_test_acc = session.run([teacher_accuracy,
                                               sgld_test_accuracy], {
            mode: 'Test',
            input_x: x_test,
            input_y: y_test
        })
        train_acc, ce = session.run([teacher_accuracy, cross_entropy], {
            mode: 'Test',
            input_x: x_train,
            input_y: y_train
        })



        if train_acc > 0.9:
            start_langevin = True

        with open(log_file_name, 'a') as f:
            f.write(LOG_DESCRIPTOR.format(k + 1, ce, train_acc, test_acc,
                                          sgld_test_acc))



        # writer.add_summary(acc_summ, global_step=k)
