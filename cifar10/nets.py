import tensorflow as tf
from LangevinSampler import RMSPropLangevin

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('train_bn', True,
                           """Either trainable BatchNorm or not""")
# tf.app.flags.DEFINE_string('optimizer', 'adam',
#                            """HA Lol""")
tf.app.flags.DEFINE_float('T', 2.,
                           """Temperature""")
tf.app.flags.DEFINE_bool('langevin', False,
                         """ Use SGLD """)
tf.app.flags.DEFINE_float('noise_ratio', 2.,
                          """ noise-variance / learning-rate in SGLD """)



def get_modelf(name):
    if name == 'vgg16-like':
        return vgg16
    else:
        raise NotImplementedError('No such model "{}"'.format(name))

def train_op(loss, var, global_step=None, learning_rate=0.001):
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        return opt.minimize(loss, var_list=var, global_step=global_step)


def t_train_op(loss, var, global_step=None, learning_rate=0.001):
    reg_loss = tf.constant(0.)

    if FLAGS.langevin:
        opt = RMSPropLangevin(learning_rate=learning_rate,
                              noise_ratio=FLAGS.noise_ratio)
        for v in var:
            reg_loss += tf.nn.l2_loss(v)
    else:
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        return opt.minimize(loss + reg_loss * FLAGS.reg, var_list=var, global_step=global_step)

def teacher_cl_loss(labels, logits):
    return tf.reduce_mean(tf.losses.softmax_cross_entropy(labels, logits))

def student_cl_loss(t_logits, s_logits):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(t_logits/FLAGS.T),
                                                logits=s_logits/FLAGS.T))

def conv_bn(net, is_training, filters, kernel_size, name=None,
            use_dropout=True):
    net = tf.layers.conv2d(net, int(filters), kernel_size, name=name)
    net = tf.layers.batch_normalization(net, training=is_training,
                                        trainable=FLAGS.train_bn, momentum=0.9)
    net = tf.nn.relu(net)

    if use_dropout:
        net = tf.layers.dropout(net, rate=0.4, training=is_training,
                                name=name + '_dropout')

    return net

def vgg16(inputs, is_training=True, nclasses=10, use_dropout=True, k=1.):
    # net = tf.contrib.layers.repeat(inputs, 2, conv_bn, is_training, 64, 3, name='conv1')
    net = conv_bn(inputs, is_training, 64 * k, 3, name='conv1_1',
                  use_dropout=use_dropout)
    net = conv_bn(net, is_training, 64 * k, 3, name='conv1_2',
                  use_dropout=False)
    net = tf.layers.max_pooling2d(net, [2, 2], 1, name='pool1')

    # net = tf.contrib.layers.repeat(net, 2, conv_bn, is_training, 128, 3, name='conv2')
    net = conv_bn(net, is_training, 128 * k, 3, name='conv2_1',
                  use_dropout=use_dropout)
    net = conv_bn(net, is_training, 128 * k, 3, name='conv2_2',
                  use_dropout=False)
    net = tf.layers.max_pooling2d(net, [2, 2], 1, name='pool2')

    # net = tf.contrib.layers.repeat(net, 3, conv_bn, is_training, 256, 3, name='conv3')
    net = conv_bn(net, is_training, 256 * k, 3, name='conv3_1',
                  use_dropout=use_dropout)
    net = conv_bn(net, is_training, 256 * k, 3, name='conv3_2',
                  use_dropout=use_dropout)
    net = conv_bn(net, is_training, 256 * k, 3, name='conv3_3',
                  use_dropout=False)
    net = tf.layers.max_pooling2d(net, [2, 2], 1, name='pool3')

    # net = tf.contrib.layers.repeat(net, 3, conv_bn, is_training, 512, 3, name='conv4')
    net = conv_bn(net, is_training, 512 * k, 3, name='conv4_1',
                  use_dropout=use_dropout)
    net = conv_bn(net, is_training, 512 * k, 3, name='conv4_2',
                  use_dropout=use_dropout)
    net = conv_bn(net, is_training, 512 * k, 3, name='conv4_3',
                  use_dropout=False)
    net = tf.layers.max_pooling2d(net, [2, 2], 1, name='pool4')

    # net = tf.contrib.layers.repeat(net, 3, conv_bn, is_training, 512, 3, name='conv5')
    net = conv_bn(net, is_training, 512 * k, 3, name='conv5_1',
                  use_dropout=use_dropout)
    net = conv_bn(net, is_training, 512 * k, 3, name='conv5_2',
                  use_dropout=use_dropout)
    net = conv_bn(net, is_training, 512 * k, 3, name='conv5_3',
                  use_dropout=False)
    net = tf.layers.max_pooling2d(net, [2, 2], 1, name='pool5')

    if use_dropout:
        net = tf.layers.dropout(net, rate=0.5, training=is_training,
                                name='dropout_before_fc6')

    net = tf.layers.dense(net, int(512 * k), name='fc6', activation=None)
    net = tf.layers.batch_normalization(net, training=is_training,
                                        trainable=FLAGS.train_bn, momentum=0.9)
    net = tf.nn.relu(net)
    if use_dropout:
        net = tf.layers.dropout(net, rate=0.5, training=is_training,
                                name='dropout_fc6')

    net = tf.layers.dense(net, nclasses, activation=None, name='logits7')

    return tf.squeeze(net)


def loss(logits, labels):
   labels = tf.cast(labels, tf.int64)
   cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
       labels=labels, logits=logits, name='cross_entropy_per_example')
   cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
   tf.add_to_collection('losses', cross_entropy_mean)

   # The total loss is defined as the cross entropy loss plus all of the weight
   # decay terms (L2 loss).
   return tf.add_n(tf.get_collection('losses'), name='total_loss')
