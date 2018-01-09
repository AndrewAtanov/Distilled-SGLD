from __future__ import absolute_import
import tensorflow as tf
from skimage.transform import rotate
from scipy.ndimage.interpolation import shift
import numpy as np
import load_data
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

def rand_rotate(img):
    sign = np.random.choice([1, -1])
    return rotate(img,sign * np.random.rand() * 60 )

def rand_shift(img):
    return shift(img, (np.random.randint(-2, 3), np.random.randint(-2, 3)))

def random_transform(img):
    return rand_rotate(rand_shift(img))

def mnist_augmentation(X, factor=1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    augment_x = []
    for img in np.squeeze(X).reshape((-1, 28, 28)):
        for _ in range(factor):
            augment_x.append(random_transform(img))

    return np.array(augment_x).reshape((-1, 28, 28, 1))

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

def batch_iterator(X, y, batch_size, epochs=1):
    for e in range(epochs):
        for i in range(0, len(X), batch_size):
            yield X[i: i + batch_size], y[i: i + batch_size]


def reduce_mean_acc(data, y, logits, bs=100):
    cumm = 0.
    n_elements = 0
    for X_batch, y_batch in load_data.iterate_minibatches(data, y, bs):
        n_elements += X_batch.shape[0]
        cumm += np.sum(logits(X_batch).argmax(axis=1) == y_batch.argmax(axis=1))

    return cumm / n_elements
