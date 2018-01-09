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


parser = argparse.ArgumentParser(description='Print id from database.csv and probability of object on image')
parser.add_argument('states', metavar='N', type=int, nargs='+', help='path to image')
parser.add_argument('--iter', default=int(5e5), type=int)
parser.add_argument('--nruns', default=10, type=int)

args = parser.parse_args()

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

def decay_lr(t, init, factor, time):
    return init * (factor**(t // time))


X, y = load_boston(return_X_y=True)

X = StandardScaler().fit_transform(X)

X, x_test, y, y_test = train_test_split(X, y, test_size=50, random_state=42)

tf.reset_default_graph()

lambda_n = tf.to_double(1.25)
_lambda = tf.to_double(1.)

input_x = tf.placeholder(tf.float64, shape=[None, X.shape[1]], name='batch')
input_y = tf.placeholder(tf.float64, [None], name='labels')
sigma_y = tf.placeholder_with_default(tf.sqrt(1/lambda_n), [])

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

likelihood_distribution = tf.contrib.distributions.Normal(teacher_mu, tf.sqrt(1/lambda_n))
prior_distribution = tf.contrib.distributions.Normal(tf.to_double(0.), tf.sqrt(1 / _lambda))
prior = tf.add_n([tf.reduce_sum(prior_distribution.log_pdf(var)) for var in tf.get_collection('Teacher-prior')])
meanlikelihood = tf.reduce_mean(likelihood_distribution.log_pdf(input_y))
posterior = meanlikelihood * tf.to_double(X.shape[0]) + prior

learning_rate = tf.placeholder(tf.float64)
opt = tf.train.GradientDescentOptimizer(learning_rate / 2)
teacher_train_op = opt.minimize(-posterior)

teacher_lr = lambda t: 1e-6

teacher_batch_size = 1

T_start = 0
T = args.iter

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

session = tf.Session(config=config)

train_llh = []
test_llh = []
test_rmse = []
train_rmse = []

for random_state in args.states:
    X, y = load_boston(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    X, x_test, y, y_test = train_test_split(X, y, test_size=50, random_state=random_state)
    train_llh.append([])
    test_llh.append([])
    test_rmse.append([])
    train_rmse.append([])

    for n_run in range(args.nruns):
        y = np.load('boston-avg-train-target_{}_{}.npy'.format(random_state, n_run))
        session.run(tf.global_variables_initializer())
        log_file_name = 'boston_SGD_DN_{}_{}'.format(random_state, n_run)
        with open(log_file_name, 'w') as f:
            f.write('#{} random_state, {} run\niteration,posterior,train_llh,train_rmse,test_llh,test_rmse,best_train_llh,best_test_llh\n'.format(random_state, n_run))
        for t in range(T_start, T_start + T):
            s = t % X.shape[0]
            session.run(teacher_train_op, {
                input_x: X[s: s + teacher_batch_size],
                input_y: y[s: s + teacher_batch_size],
                learning_rate: teacher_lr(t + 1)
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

                with open(log_file_name, 'a') as f:
                    f.write('{},{},{},{},{},{},{},{}\n'.format(t+1, t_posterior, t_train_llh,
                                                               t_train_rmse, t_test_llh, t_test_rmse,
                                                               t_train_llh_best_sigma, t_test_llh_best_sigma))
        # train_llh[-1].append(session.run(meanlikelihood, {input_x: X, input_y: y}))
        # test_llh[-1].append(session.run(meanlikelihood, {input_x: x_test, input_y: y_test}))
        # test_rmse[-1].append(np.sqrt(mean_squared_error(session.run(teacher_mu, {input_x: x_test}), y_test)))
        # train_rmse[-1].append(np.sqrt(mean_squared_error(session.run(teacher_mu, {input_x: X}), y)))

        # print(random_state, n_run)

# np.save('train_llh.npy', train_llh)
# np.save('test_llh.npy', test_llh)
# np.save('test_rmse.npy', test_rmse)
# np.save('train_rmse.npy', train_rmse)
