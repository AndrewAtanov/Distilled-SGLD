import numpy as np
import tensorflow as tf
from sklearn.metrics import *
import sys
import os
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from multiprocessing import Process, Semaphore, Lock
import argparse
from scipy.stats import norm

LOG_START = "iteration,posterior,KL,"\
            "t_train_llh,t_train_rmse,t_test_llh,t_test_rmse,t_best_train_llh,t_best_test_llh,"\
            "s_train_llh,s_train_rmse,s_test_llh,s_test_rmse,s_best_train_llh,s_best_test_llh\n"

LOG_DESCRIPTOR = "{},"*14 + "{}\n"

def nofc(collection):
    return [x.name for x in collection]

def get_all_variables_from_scope(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

def get_names(variables):
    return [x.name for x in variables]

def get_variable_from_scope(scope, var, dtype):
    with tf.variable_scope(scope, reuse=True):
        return tf.get_variable(name=var, dtype=dtype)

def choose_tensors_from_scope(tensors, scope):
    return [x for x in tensors if x.name.startswith(scope)]

class LangevinSampler(tf.train.GradientDescentOptimizer):
    def __init__(self, learning_rate, noise_rate):
        self.noise_rate = noise_rate
        super(LangevinSampler, self).__init__(learning_rate)

    def sample(self, target, var_list=None, return_vars=True):
        minimize_op = self.minimize(-target, var_list=var_list)
        # return minimize_op, None
        if var_list is None:
            var_list = tf.trainable_variables() + ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        vars_noises = [(var, tf.random_normal(var.get_shape(), stddev=tf.sqrt(self.noise_rate), dtype=tf.float64)) for var in var_list]
        with tf.control_dependencies([minimize_op]):
            langevin_step = tf.group(*[tf.assign_add(var, noise) for var, noise in vars_noises])

        if not return_vars:
            return minimize_op, langevin_step

        with tf.control_dependencies([langevin_step]):
            get_vars = tf.identity(var_list)

        return get_vars

parser = argparse.ArgumentParser(description='Print id from database.csv and probability of object on image')
parser.add_argument('states', metavar='N', type=int, nargs='+', help='path to image')
parser.add_argument('--lambda_n', default=1.25, type=float)
parser.add_argument('--lamb', default=2.5, type=float)
parser.add_argument('--gamma', default=1e-3, type=float)
parser.add_argument('--iter', default=int(5e5), type=int)
args = parser.parse_args()

X, y = load_boston(return_X_y=True)
X = StandardScaler().fit_transform(X)
X, x_test, y, y_test = train_test_split(X, y, test_size=50, random_state=42)


tf.reset_default_graph()

input_x = tf.placeholder(tf.float64, shape=[None, X.shape[1]], name='batch')
input_y = tf.placeholder(tf.float64, [None], name='labels')

lambda_n = tf.to_double(args.lambda_n)
_lambda = tf.to_double(args.lamb)
gamma = tf.to_double(args.gamma)

with tf.variable_scope('Teacher'):
    teacher_hidden = tf.contrib.layers.fully_connected(input_x, 50,
                                                       variables_collections={'weights': ['Teacher-prior'],
                                                                              'biases': ['Teacher-b']},
                                                       weights_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                         maxval=1.))
    teacher_mu = tf.contrib.layers.fully_connected(teacher_hidden, 1,
                                                   variables_collections={'weights': ['Teacher-prior'],
                                                                          'biases': ['Teacher-b']},
                                                   weights_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                     maxval=1.),
                                                   activation_fn=None)[:,0]

with tf.variable_scope('Student'):
    student_hidden = tf.contrib.layers.fully_connected(input_x, 50, weights_regularizer=tf.nn.l2_loss)
    student_out = tf.contrib.layers.fully_connected(student_hidden, 2, weights_regularizer=tf.nn.l2_loss)
    student_mu = student_out[:,0]
    student_logsigma = student_out[:,1]

    student_reg = gamma * tf.add_n(choose_tensors_from_scope(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), 'Student'))
    student_loss = tf.reduce_mean(student_logsigma + tf.exp(-student_logsigma) * ( (teacher_mu - student_mu)**2 + 1 / lambda_n)) / 2 + student_reg

likelihood_distribution = tf.contrib.distributions.Normal(teacher_mu,
                                                          tf.sqrt(1 / lambda_n))
prior_distribution = tf.contrib.distributions.Normal(tf.to_double(0.),
                                                     tf.sqrt(1 / _lambda))
prior = tf.add_n([tf.reduce_sum(prior_distribution.log_pdf(var)) for var in tf.get_collection('Teacher-prior')])
meanlikelihood = tf.reduce_mean(likelihood_distribution.log_pdf(input_y))
posterior = meanlikelihood * tf.to_double(X.shape[0]) + prior

student_likelihood_dist = tf.contrib.distributions.Normal(student_mu, tf.sqrt(1 / lambda_n))
student_mean_likelihood = tf.reduce_mean(student_likelihood_dist.log_pdf(input_y))

learning_rate = tf.placeholder(tf.float64)
teacher_sampler = LangevinSampler(learning_rate / tf.to_double(2.), learning_rate)
teacher_train_op, langevin_step = teacher_sampler.sample(posterior, var_list=get_all_variables_from_scope('Teacher'), return_vars=False)
# opt = tf.train.GradientDescentOptimizer(learning_rate / 2)
# teacher_train_op = opt.minimize(-posterior)


slr = tf.placeholder(tf.float64)
student_optimizer = tf.train.AdamOptimizer()
student_train_op = student_optimizer.minimize(student_loss, var_list=get_all_variables_from_scope('Student'))


def decay_lr(t, init, factor, time):
    return init * (factor**(t // time))

teacher_lr = lambda t: decay_lr(t, 1e-5, 0.5, 80000)
student_lr = lambda t: decay_lr(t, 1e-2, 0.8, 5000)

def student_data(x, factor=1):
    return np.concatenate([x + np.random.normal(scale=0.05, size=x.shape) for _ in range(factor)], axis=0)

teacher_batch_size = 1
student_factor_size = 10

T_start = 0
T = args.iter

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
session = tf.Session(config=config)

for random_state in args.states:

    X, y = load_boston(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X, x_test, y, y_test = train_test_split(X, y, test_size=50, random_state=random_state)

    for n_run in range(10):
        session.run(tf.global_variables_initializer())

        kl = 0
        kl_facor = 0.995

        log_file_name = 'boston_distilled_SGLD_{}_{}'.format(random_state, n_run)
        with open(log_file_name, 'w') as f:
            f.write(LOG_START)

        for t in range(T):
            s = t % X.shape[0]
            session.run(langevin_step, {
                input_x: X[s: s + teacher_batch_size],
                input_y: y[s: s + teacher_batch_size],
                learning_rate: teacher_lr(t + 1),
            })

            student_batch = student_data(X[s: s + teacher_batch_size],
                                         factor=student_factor_size)
            session.run(student_train_op, {
                input_x: student_batch,
                slr: student_lr(t + 1)
            })


            kl = kl * kl_facor + (1 - kl_facor) * session.run(student_loss, {
                input_x: student_batch,
                })

            if (t+1) % 1000 == 0:
                t_test_mu = session.run(teacher_mu, {input_x: x_test})
                b_std = np.sqrt(np.mean((t_test_mu - y_test)**2))
                t_test_llh_best_sigma = np.mean(norm.logpdf(t_test_mu - y_test, scale=b_std))

                t_train_mu = session.run(teacher_mu, {input_x: X})
                b_std = np.sqrt(np.mean((t_train_mu - y)**2))
                t_train_llh_best_sigma = np.mean(norm.logpdf(t_train_mu - y, scale=b_std))

                t_posterior = session.run(posterior, {input_x: X, input_y: y})
                t_test_llh = session.run(meanlikelihood, {input_x: x_test, input_y: y_test})
                t_train_llh = session.run(meanlikelihood, {input_x: X, input_y: y})
                t_train_rmse = np.sqrt(mean_squared_error(session.run(teacher_mu, {input_x: X}), y))
                t_test_rmse = np.sqrt(mean_squared_error(session.run(teacher_mu, {input_x: x_test}), y_test))

                s_test_mu = session.run(student_mu, {input_x: x_test})
                b_std = np.sqrt(np.mean((s_test_mu - y_test)**2))
                s_test_llh_best_sigma = np.mean(norm.logpdf(s_test_mu - y_test, scale=b_std))

                s_train_mu = session.run(student_mu, {input_x: X})
                b_std = np.sqrt(np.mean((s_train_mu - y)**2))
                s_train_llh_best_sigma = np.mean(norm.logpdf(s_train_mu - y, scale=b_std))

                s_test_llh = session.run(student_mean_likelihood, {input_x: x_test, input_y: y_test})
                s_train_llh = session.run(student_mean_likelihood, {input_x: X, input_y: y})
                s_train_rmse = np.sqrt(mean_squared_error(session.run(student_mu, {input_x: X}), y))
                s_test_rmse = np.sqrt(mean_squared_error(session.run(student_mu, {input_x: x_test}), y_test))


                with open(log_file_name, 'a') as f:
                    f.write(LOG_DESCRIPTOR.format(t+1, t_posterior, kl,
                                                  t_train_llh, t_train_rmse,
                                                  t_test_llh, t_test_rmse,
                                                  t_train_llh_best_sigma, t_test_llh_best_sigma,
                                                  s_train_llh, s_train_rmse,
                                                  s_test_llh, s_test_rmse,
                                                  s_train_llh_best_sigma, s_test_llh_best_sigma))
