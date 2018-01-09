import tensorflow as tf


def mnist_fc(input_x, collection, activation_fn, keep_prob, is_training):
    variables_collections = {
        'weights': [collection + '-w'],
        'biases': [collection + '-b']
    }
    h = tf.contrib.layers.fully_connected(input_x, 300,
                                          weights_initializer=tf.random_normal_initializer(),
                                          variables_collections=variables_collections,
                                          activation_fn=activation_fn)
    h = tf.contrib.layers.dropout(inputs=h, keep_prob=keep_prob,
                                  is_training=is_training)

    h = tf.contrib.layers.fully_connected(input_x, 100,
                                          weights_initializer=tf.random_normal_initializer(),
                                          variables_collections=variables_collections,
                                          activation_fn=activation_fn)
    h = tf.contrib.layers.dropout(inputs=h, keep_prob=keep_prob,
                                  is_training=is_training)

    logits = tf.contrib.layers.fully_connected(h, 10,
                                               weights_initializer=tf.random_normal_initializer(),
                                               variables_collections=variables_collections,
                                               activation_fn=None)
    return logits

def mnist_lenet5(input_x, collection, activation_fn, keep_prob, is_training):
    variables_collections = {
        'weights': [collection + '-w'],
        'biases': [collection + '-b']
    }
    h = tf.contrib.layers.convolution2d(input_x, num_outputs=20,
                                        kernel_size=5,
                                        stride=1,
                                        activation_fn=tf.nn.relu,
                                        variables_collections=variables_collections,
                                        weights_initializer=tf.random_normal_initializer())
    h = tf.contrib.layers.max_pool2d(h, kernel_size=2)
    h = tf.contrib.layers.dropout(inputs=h, keep_prob=keep_prob,
                                  is_training=is_training)

    h = tf.contrib.layers.convolution2d(h, num_outputs=50,
                                        kernel_size=5,
                                        stride=1,
                                        activation_fn=tf.nn.relu,
                                        variables_collections=variables_collections,
                                        weights_initializer=tf.random_normal_initializer())
    h = tf.contrib.layers.max_pool2d(h, kernel_size=2)
    h = tf.contrib.layers.dropout(inputs=h, keep_prob=keep_prob,
                                  is_training=is_training)

    h = tf.contrib.layers.flatten(h)
    h = tf.contrib.layers.fully_connected(h, 500,
                                          weights_initializer=tf.random_normal_initializer(),
                                          variables_collections=variables_collections,
                                          activation_fn=activation_fn)
    h = tf.contrib.layers.dropout(inputs=h, keep_prob=keep_prob,
                                  is_training=is_training)

    logits = tf.contrib.layers.fully_connected(h, 10,
                                          weights_initializer=tf.random_normal_initializer(),
                                          variables_collections=variables_collections,
                                          activation_fn=None)

    return logits
