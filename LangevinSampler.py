import tensorflow as tf

class LangevinSampler(tf.train.GradientDescentOptimizer):
    def __init__(self, learning_rate, noise_rate):
        self.noise_rate = noise_rate
        super(LangevinSampler, self).__init__(learning_rate)

    def sample(self, target, var_list=None, return_vars=False):
        minimize_op = self.minimize(-target, var_list=var_list)
        # return minimize_op, None
        if var_list is None:
            var_list = tf.trainable_variables() + ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        vars_noises = [(var, tf.random_normal(var.get_shape(), stddev=tf.sqrt(self.noise_rate), dtype=tf.float32)) for var in var_list]
        with tf.control_dependencies([minimize_op]):
            langevin_step = tf.group(*[tf.assign_add(var, noise) for var, noise in vars_noises])

        if not return_vars:
            return minimize_op, langevin_step

        with tf.control_dependencies([langevin_step]):
            get_vars = tf.identity(var_list)

        return get_vars


class RMSPropLangevin(tf.train.RMSPropOptimizer):
    def __init__(self, learning_rate, noise_ratio):
        self._noise_ratio = noise_ratio
        self._learning_rate = learning_rate
        super(RMSPropLangevin, self).__init__(learning_rate)

    def sample(self, target, var_list=None):
        minimize_op = self.minimize(target, var_list=var_list)
        if var_list is None:
            var_list = tf.trainable_variables() + ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)

        langevin_upd = []
        for var in var_list:
            lr = self._learning_rate_tensor / tf.sqrt(self.get_slot(var, "rms") + self._epsilon_tensor)
            noise = tf.random_normal(tf.shape(var), stddev=lr * tf.cast(self._noise_ratio, lr.dtype))
            langevin_upd.append(tf.assign_add(var, noise))

        with tf.control_dependencies([minimize_op]):
            langevin_step = tf.group(*langevin_upd)

        return minimize_op, langevin_step
