import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from mnist_nets import mnist_lenet5
from LangevinSampler import LangevinSampler, RMSPropLangevin
from utils import *
import os
from itertools import count
import argparse
import sys


session_config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
# please do not use the totality of the GPU memory
session_config.gpu_options.per_process_gpu_memory_fraction = 0.30

def compute_accuracy(sess, acc_op, X, y, batch_size=100):
    acc = 0
    for _x, _y in batch_iterator(X, y, batch_size=batch_size):
        acc += sess.run(acc_op, {
            input_x: _x,
            input_y: _y,
            mode: 'Test',
            T: 1.
        }) * float(len(_y))

    return acc / float(len(y))


LOG_START = 'iteration,cross_entropy,train_accuracy,test_accuracy,sgld_test_accuracy,KL,s_train_accuracy,s_test_accuracy\n'
LOG_DESCRIPTOR = '{},'*sum([1 for x in LOG_START if x == ',']) + '{}\n'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', default='~/atanov/data/MNIST_data/')
parser.add_argument('--summ_dir', default=None)
parser.add_argument('--t_keep_prob', default=1., type=float)
parser.add_argument('--s_keep_prob', default=1., type=float)
parser.add_argument('--epoch', default=int(1000), type=int)
parser.add_argument('--log', default=int(100), type=int)
parser.add_argument('--burn', default=int(1000), type=int)
parser.add_argument('--tbs', default=int(10), type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--noise_ratio', default=2, type=float)
parser.add_argument('--aug_factor', default=None, type=int)
parser.add_argument('-T', default=1e-2, type=float)
args = parser.parse_args()

args.pid = os.getpid()

log_file_name = None

EXP_INFO = '# "t_keep_prob" = {t_keep_prob}, "s_keep_prob" = {s_keep_prob}, "batch_size" = {tbs}, "lr" = {lr}, "noise_ratio": {noise_ratio}, "burn": {burn}, "aug_factor": {aug_factor}, "": {}\n'

if args.summ_dir is None:
    for i in count():
        name = 'mnist-LeNet-SGLD-{}'.format(i)
        if not os.path.exists(name):
            # os.makedirs('./{}'.format(i))
            # args.summ_dir = './{}'.format(i)
            log_file_name = name
            with open(name, 'w') as f:
                f.write('#' + str(vars(args)) + '\n')
                f.write(LOG_START)
            break


mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
x_train = np.concatenate((mnist.train.images, mnist.validation.images), axis=0).reshape((-1, 28, 28, 1))
y_train = np.concatenate((mnist.train.labels, mnist.validation.labels), axis=0)

x_test = mnist.test.images.copy().reshape((-1, 28, 28, 1))
y_test = mnist.test.labels.copy()

T = tf.placeholder_with_default(args.T, [])

# train_mdoe = tf.placeholder_with_default(True, [])
mode = tf.placeholder(tf.string, [])
input_x = tf.placeholder(tf.float32, [None, 28, 28, 1])
input_y = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope('Teacher'):
    t_logits = mnist_lenet5(input_x, 'Teacher', tf.nn.relu,
                          args.t_keep_prob, tf.equal(mode, 'Train'))
    t_proba = tf.nn.softmax(t_logits)
    t_cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=input_y,
                                                logits=t_logits))

with tf.variable_scope('Student'):
    s_logits = mnist_lenet5(input_x, 'Student', tf.nn.relu,
                            args.s_keep_prob, tf.equal(mode, 'Train'))
    s_proba = tf.nn.softmax(s_logits)
    s_cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(t_logits/T),
                                                logits=s_logits/T))

print(t_logits.dtype, input_y.dtype)

n_models = tf.get_variable('n_models', shape=[], initializer=tf.constant_initializer(0, dtype=tf.int32),
                           dtype=tf.int32)
test_avg_pred = tf.get_variable('test_avg_pred', shape=y_test.shape,
                                initializer=tf.zeros_initializer())

with tf.control_dependencies([tf.assign_add(n_models, tf.to_int32(1))]):
    upd_test_pred = tf.assign(test_avg_pred,
                              (test_avg_pred * tf.to_float(n_models - 1) + t_proba) / tf.to_float(n_models))

sgld_test_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(test_avg_pred, 1),
                                                         tf.argmax(input_y, 1))))

teacher_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(t_logits, 1),
                                                       tf.argmax(input_y, 1))))

student_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(s_logits, 1),
                                                       tf.argmax(input_y, 1))))

optimizer = RMSPropLangevin(args.lr, tf.to_double(args.noise_ratio))
teacher_train_op, langevin_step = optimizer.sample(t_cross_entropy,
                                                   var_list=get_all_variables_from_scope('Teacher'))


s_log_proba = tf.log(s_proba)
student_loss = s_cross_entropy
student_optimizer = tf.train.AdamOptimizer()
student_train_op = student_optimizer.minimize(student_loss,
                                              var_list=get_all_variables_from_scope('Student'))

cross_entropy_summary = tf.summary.scalar('cross_entropy', t_cross_entropy)
train_acc_summ = tf.summary.scalar('Train', teacher_accuracy)
test_acc_summ = tf.summary.scalar('Test', teacher_accuracy)


session = tf.Session(config=session_config)
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

    if k > 100:
        s_batch = batch_xs
        if args.aug_factor:
            s_batch = np.concatenate((mnist_augmentation(batch_xs, factor=args.aug_factor),
                                      s_batch), axis=0)

        session.run(student_train_op, {
            input_x: s_batch,
            mode: 'Train'
        })

    if (k + 1) % args.burn == 0:
        session.run(upd_test_pred, {
            mode: 'Test',
            input_x: x_test,
        })

    if (k + 1) % args.log == 0:
        test_acc = 0
        train_acc = 0
        sgld_test_acc = 0
        s_test_acc = 0

        sgld_predict = np.argmax(session.run(test_avg_pred), axis=1)
        sgld_test_acc = np.mean(np.argmax(y_test, axis=1) == sgld_predict)

        test_acc = compute_accuracy(session, teacher_accuracy,
                                    x_test, y_test, batch_size=100)
        train_acc = compute_accuracy(session, teacher_accuracy,
                                     x_train, y_train, batch_size=100)
        s_test_acc = compute_accuracy(session, student_accuracy,
                                      x_test, y_test, batch_size=100)

        ce = 0
        s_train_acc, kl = 0, 0

        if train_acc > 0.9:
            start_langevin = True

        with open(log_file_name, 'a') as f:
            f.write(LOG_DESCRIPTOR.format(k + 1, ce,
                                          train_acc, test_acc, sgld_test_acc,
                                          kl, s_train_acc, s_test_acc))



        # writer.add_summary(acc_summ, global_step=k)
