import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from mnist_nets import mnist_lenet5
from LangevinSampler import LangevinSampler, RMSPropLangevin
from utils import *
import os
from itertools import count
import argparse

LOG_START = 'iteration,cross_entropy,train_accuracy,test_accuracy\n'
LOG_DESCRIPTOR = '{},'*3 + '{}\n'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', default='~/atanov/data/MNIST_data/')
parser.add_argument('--summ_dir', default=None)
parser.add_argument('--keep_prob', default=0.5, type=float)
parser.add_argument('--epoch', default=int(100), type=int)
parser.add_argument('--log', default=int(100), type=int)
parser.add_argument('--tbs', default=int(10), type=int)
parser.add_argument('--lr', default=0.01, type=float)
args = parser.parse_args()

log_file_name = None

# EXP_INFO = "{'keep_prob': {keep_prob}, 't_batch_size' = {tbs},}"

if args.summ_dir is None:
    for i in count():
        name = 'mnist-LeNet-SGD-{}'.format(i)
        if not os.path.exists(name):
            # os.makedirs('./{}'.format(i))
            # args.summ_dir = './{}'.format(i)
            log_file_name = name
            with open(name, 'w') as f:
                f.write('# "keep_prob" = {keep_prob}, "batch_size" = {tbs}, "lr" = {lr}\n'.format(**vars(args)))
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

teacher_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, 1),
                                                       tf.argmax(input_y, 1))))

optimizer = RMSPropLangevin(args.lr, tf.to_double(2.))
teacher_train_op, langevin_step = optimizer.sample(cross_entropy,
                                                   var_list=get_all_variables_from_scope('Teacher'))

# optimizer = tf.train.RMSPropOptimizer(args.lr)
# teacher_train_op = optimizer.minimize(cross_entropy,
#                                       var_list=get_all_variables_from_scope('Teacher'))

cross_entropy_summary = tf.summary.scalar('cross_entropy', cross_entropy)
train_acc_summ = tf.summary.scalar('Train', teacher_accuracy)
test_acc_summ = tf.summary.scalar('Test', teacher_accuracy)



session = tf.Session()
session.run(tf.global_variables_initializer())

# writer = tf.summary.FileWriter(args.summ_dir, session.graph)
def decay_lr(t, init, factor, time):
    return init * (factor**(t // time))

teacher_lr_policy = lambda t: 1. / t

teacher_batch_size = args.tbs

for k, [batch_xs, batch_ys] in enumerate(batch_iterator(x_train, y_train,
                                                        args.tbs,
                                                        epochs=args.epoch)):
    _, ce_summ, acc_summ = session.run([teacher_train_op, cross_entropy_summary,
                                        train_acc_summ], {
        input_x: batch_xs,
        input_y: batch_ys,
        mode: 'Train',
    })

    # writer.add_summary(ce_summ, global_step=k)
    # writer.add_summary(acc_summ, global_step=k)

    if (k + 1) % args.log == 0:
        test_acc = session.run(teacher_accuracy, {
            mode: 'Test',
            input_x: x_test,
            input_y: y_test
        })
        train_acc, ce = session.run([teacher_accuracy, cross_entropy], {
            mode: 'Test',
            input_x: x_train,
            input_y: y_train
        })

        with open(log_file_name, 'a') as f:
            f.write(LOG_DESCRIPTOR.format(k + 1, ce, train_acc, test_acc))



        # writer.add_summary(acc_summ, global_step=k)
