import numpy as np
import tensorflow as tf
from sklearn.metrics import *

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_boston(return_X_y=True)
X, x_test, y, y_test = train_test_split(X, y, test_size=50, random_state=42)

X = StandardScaler().fit_transform(X)


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

tf.reset_default_graph()

beta = 1.25
alpha = 5

input_x = tf.placeholder(tf.float64, shape=[None, X.shape[1]], name='batch')
# student_input_x = tf.placeholder(tf.float64, shape=[None, X.shape[1]], name='student-batch')
input_y = tf.placeholder(tf.float64, [None], name='labels')
sigma_y = tf.placeholder_with_default(tf.cast(1/beta, tf.float64), [])
# alpha = tf.placeholder_with_default(tf.cast(1., tf.float64), [])
lambda_n = tf.to_double(beta)
gamma = tf.to_double(1e-2)

with tf.variable_scope('Teacher'):
    teacher_hidden = tf.contrib.layers.fully_connected(input_x, 50, variables_collections={'weights': ['Teacher-prior'], 'biases': ['Teacher-b']})
    teacher_mu = tf.contrib.layers.fully_connected(teacher_hidden, 1, variables_collections={'weights': ['Teacher-prior'], 'biases': ['Teacher-b']}, activation_fn=None)[:,0]

with tf.variable_scope('Student'):
    student_hidden = tf.contrib.layers.fully_connected(input_x, 50, weights_regularizer=tf.nn.l2_loss)
    student_out = tf.contrib.layers.fully_connected(student_hidden, 2, weights_regularizer=tf.nn.l2_loss)
    student_mu = student_out[:,0]
    student_logsigma = student_out[:,1]

    student_reg = gamma * tf.add_n(choose_tensors_from_scope(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), 'Student'))
    student_loss = tf.reduce_mean(student_logsigma + tf.exp(-student_logsigma) * ( (teacher_mu - student_mu)**2 + 1 / lambda_n)) / 2 + student_reg

likelihood_distribution = tf.contrib.distributions.Normal(teacher_mu, sigma_y)
prior_distribution = tf.contrib.distributions.Normal(tf.to_double(0.), tf.to_double(1 / alpha))
prior = tf.add_n([tf.reduce_sum(prior_distribution.log_pdf(var)) for var in tf.get_collection('Teacher-prior')])
meanlikelihood = tf.reduce_mean(likelihood_distribution.log_pdf(input_y))
posterior = meanlikelihood * tf.to_double(X.shape[0]) + prior

student_likelihood_dist1 = tf.contrib.distributions.Normal(student_mu, tf.exp(student_logsigma))
student_likelihood_dist2 = tf.contrib.distributions.Normal(student_mu, 1/lambda_n)


student_mean_likelihood1 = tf.reduce_mean(student_likelihood_dist1.log_pdf(input_y))
student_mean_likelihood2 = tf.reduce_mean(student_likelihood_dist2.log_pdf(input_y))


learning_rate = tf.placeholder(tf.float64)
teacher_sampler = LangevinSampler(learning_rate, learning_rate)
teacher_train_op, langevin_step = teacher_sampler.sample(posterior, var_list=get_all_variables_from_scope('Teacher'), return_vars=False)

slr = tf.placeholder(tf.float64)
student_optimizer = tf.train.AdamOptimizer()
student_train_op = student_optimizer.minimize(student_loss, var_list=get_all_variables_from_scope('Student'))

teacher_batch_size = 1
student_batch_size = 10

T_start = 0
T = int(1e6)

teacher_x = X + np.random.normal(size=X.shape)

for t in range(T_start, T_start + T):
    indicies = np.random.choice(X.shape[0], size=teacher_batch_size, replace=False)
    s = t % X.shape[0]
    session.run(langevin_step, {
        input_x: X[s: s + teacher_batch_size],
        input_y: y[s: s + teacher_batch_size],
        learning_rate: teacher_lr(t + 1)
    })

    teacher_losses.append(-session.run(posterior, {
        input_x: X,
        input_y: y
    }))

    if (t+1) % 10000 == 0:
        print(t+1, 'iterations done')
        t_mse = mean_squared_error(y, session.run(teacher_mu, {input_x:X}))
        t_r2 = r2_score(y, session.run(teacher_mu, {input_x:X}))
        t_tr2 = r2_score(y_test, session.run(teacher_mu, {input_x: x_test}))
        s_r2 = r2_score(y, session.run(student_mu, {input_x:X}))
        s_tr2 = r2_score(y_test, session.run(student_mu, {input_x: x_test}))


        print('Teacher Train R2 {}'.format(t_r2))
        print('Teacher Test  R2 {}'.format(t_tr2))

        print('Teacher Train MSE {}'.format(t_mse))
        print('Teacher Test MeanLLH {}'.format(session.run(meanlikelihood, {input_x: x_test, input_y: y_test})))
        print('Teacher Train MeanLLH {}'.format(session.run(meanlikelihood, {input_x: X, input_y: y})))

        print('Posterior {}'.format(-teacher_losses[-1]))
