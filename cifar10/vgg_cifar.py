import tensorflow as tf
from layers import Conv2D, Dense, Conv2DVDARD, Conv2DGroupVDARD, DenseGroupVDARD, DenseVDARD, Conv2DGroupVHSppNCP, DenseGroupVHSppNCP
from datasets import CIFAR10
import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow.contrib.slim import flatten
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import time


def create_conv_bn_relu_block(conv_layer, parent, block_size, is_training, sample, init_generator, dropout=None, pool=False, N=1):
    block_size = int(block_size)
    input_shape = [None] + [s.value for s in parent.get_shape()[1:]]
    conv = conv_layer(block_size, 3, 3, N=N, input_shape=input_shape, pad=1, border_mode='VALID', logging=FLAGS.logging,
                      sparsity_rate_a=FLAGS.sa, sparsity_rate_b=FLAGS.sb, t0=FLAGS.t0,
                      W_initializer=next(init_generator), b_initializer=next(init_generator))

    if FLAGS.bn:
        conv_out = tf.layers.batch_normalization(conv(parent), training=is_training,
                                                 beta_initializer=tf.constant_initializer(next(init_generator)),
                                                 gamma_initializer=tf.constant_initializer(next(init_generator)))
    else:
        conv_out = conv(parent)
    h = tf.nn.relu(conv_out)
    if dropout is not None and FLAGS.sparsity == 'none':
        h = tf.cond(sample, lambda: tf.nn.dropout(h, keep_prob=1 - dropout), lambda: h)
    if pool:
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    return conv, h


def create_vgglike(conv_layer, fc_layer, input_x, is_training, sample, init_generator, k=1, N=1):
    layers = []

    # Decorate create_conv_bn_relu_block(...)
    def conv_bn_relu(inp, size, dropout=None, pool=False):
        l, h = create_conv_bn_relu_block(conv_layer, inp, size * k, is_training, sample, init_generator, dropout=dropout, pool=pool, N=N)
        print 'Created new conv bn relu block of size %d. Output shape:' % int(size * k), h.get_shape()
        layers.append(l)
        return h

    h = conv_bn_relu(input_x, 64, dropout=0.3)
    h = conv_bn_relu(h, 64, pool=True)

    h = conv_bn_relu(h, 128, dropout=0.4)
    h = conv_bn_relu(h, 128, pool=True)

    h = conv_bn_relu(h, 256, dropout=0.4)
    h = conv_bn_relu(h, 256, dropout=0.4)
    h = conv_bn_relu(h, 256, pool=True)

    h = conv_bn_relu(h, 512, dropout=0.4)
    h = conv_bn_relu(h, 512, dropout=0.4)
    h = conv_bn_relu(h, 512, pool=True)

    h = conv_bn_relu(h, 512, dropout=0.4)
    h = conv_bn_relu(h, 512, dropout=0.4)
    h = conv_bn_relu(h, 512, pool=True)

    h = flatten(h)
    fc = fc_layer(int(512 * k), N=N, input_dim=h.get_shape()[1].value, logging=FLAGS.logging, sparsity_rate_a=FLAGS.sa,
                  sparsity_rate_b=FLAGS.sb, t0=FLAGS.t0, W_initializer=next(init_generator), b_initializer=next(init_generator))
    if FLAGS.bn:
        fc_out = tf.layers.batch_normalization(fc(h), training=is_training,
                                               beta_initializer=tf.constant_initializer(next(init_generator)),
                                               gamma_initializer=tf.constant_initializer(next(init_generator)))
    else:
        fc_out = fc(h)
    h = tf.nn.relu(fc_out)
    if FLAGS.sparsity == 'none':
        h = tf.cond(sample, lambda: tf.nn.dropout(h, keep_prob=0.5), lambda: h)
    layers.append(fc)
    fc = fc_layer(10, N=N, input_dim=h.get_shape()[1].value, logging=FLAGS.logging, sparsity_rate_a=FLAGS.sa,
                  sparsity_rate_b=FLAGS.sb, t0=FLAGS.t0, W_initializer=next(init_generator), b_initializer=next(init_generator))
    logits = fc(h)
    layers.append(fc)

    return layers, logits


def batch_iterator(data, y, batchsize, data_augmentation=True):
    PIXELS = 32
    PAD_CROP = 4
    n_samples = data.shape[0]
    # Shuffles indicies of training data, so we can draw batches from random indicies instead of shuffling whole data
    indx = np.random.permutation(xrange(n_samples))
    for i in xrange((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[indx[sl]]
        y_batch = y[indx[sl]]

        if data_augmentation:
            # pad and crop settings
            trans_1 = np.random.randint(0, (PAD_CROP*2))
            trans_2 = np.random.randint(0, (PAD_CROP*2))
            crop_x1 = trans_1
            crop_x2 = (PIXELS + trans_1)
            crop_y1 = trans_2
            crop_y2 = (PIXELS + trans_2)

            # flip left-right choice
            flip_lr = np.random.randint(0,1)

            # set empty copy to hold augmented images so that we don't overwrite
            X_batch_aug = np.copy(X_batch)

            # for each image in the batch do the augmentation
            for j in xrange(X_batch.shape[0]):
                # for each image channel
                for k in xrange(X_batch.shape[3]):
                    # pad and crop images
                    img_pad = np.pad(
                        X_batch_aug[j, :, :, k], pad_width=((PAD_CROP, PAD_CROP), (PAD_CROP, PAD_CROP)), mode='constant')
                    X_batch_aug[j, :, :, k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                    # flip left-right if chosen
                    if flip_lr == 1:
                        X_batch_aug[j, :, :, k] = np.fliplr(X_batch_aug[j, :, :, k])
        else:
            X_batch_aug = X_batch
        yield X_batch_aug, y_batch


def main():
    if FLAGS.pretrained_model:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(FLAGS.pretrained_model + '/model.meta')
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrained_model))
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            vars_vals = sess.run(vars)
            init_generator = (x for x in vars_vals)
    else:
        init_generator = (None for _ in xrange(58))

    tf.reset_default_graph()

    layers = []
    training = tf.placeholder(tf.bool, name='training_phase')
    sample = tf.placeholder(tf.bool, name='sampling')
    use_theta = tf.placeholder(tf.bool, name='use_theta')

    mnist = CIFAR10()
    (xtrain, ytrain), (xtest, ytest) = mnist.images()
    ytrain, ytest = to_categorical(ytrain, 10), to_categorical(ytest, 10)

    N, height, width, n_channels = xtrain.shape
    iter_per_epoch = N / 100

    input_x = tf.placeholder(tf.float32, [None, height, width, n_channels], name='x')
    input_y = tf.placeholder(tf.float32, [None, 10], name='y_')

    sess = tf.InteractiveSession()

    if FLAGS.sparsity == 'none':
        fc_layer, conv_layer = Dense, Conv2D
    elif FLAGS.sparsity == 'group':
        fc_layer, conv_layer = DenseGroupVDARD, Conv2DGroupVDARD
    elif FLAGS.sparsity == 'hs++':
        fc_layer, conv_layer = DenseGroupVHSppNCP, Conv2DGroupVHSppNCP
    else:
        fc_layer, conv_layer = DenseVDARD, Conv2DVDARD

    print 'fc_layer: {}, conv_layer: {}'.format(fc_layer, conv_layer)
    with tf.variable_scope('model_construction'):
       layers, logits = create_vgglike(conv_layer, fc_layer, input_x, training, sample, init_generator, k=FLAGS.model_size,
                                       N=xtrain.shape[0])

    l2_term = 0
    for param in tf.get_collection('l2'):
        l2_term += FLAGS.l2 * 0.5 * tf.reduce_sum(tf.square(param))

    with tf.name_scope('KL_prior'):
        regs = 0
        for j, layer in enumerate(layers):
            with tf.name_scope('kl_layer{}'.format(j + 1)):
                regi = layer.get_reg()
                tf.summary.scalar('kl_layer{}'.format(j + 1), regi)
            regs += regi
        regs += l2_term
        tf.summary.scalar('KL prior', regs)

    global_step = tf.Variable(0, trainable=False)

    test_acc_placeholder = tf.placeholder(tf.float32, name='test_acc_placeholder')
    test_accuracy = tf.Variable(0., trainable=False)
    assign_test_acc = tf.assign(test_accuracy, test_acc_placeholder)

    with tf.name_scope('stats'):
        if FLAGS.anneal:
            number_zero, original_zero = FLAGS.epzero, FLAGS.epochs / 2
            with tf.name_scope('annealing_beta'):
                max_zero_step = number_zero * iter_per_epoch
                original_anneal = original_zero * iter_per_epoch
                beta_t_val = tf.cast((tf.cast(global_step, tf.float32) - max_zero_step) / original_anneal, tf.float32)
                beta_t = tf.maximum(beta_t_val, 0.)
                annealing = tf.minimum(1.,
                                       tf.cond(global_step < max_zero_step, lambda: tf.zeros((1,))[0], lambda: beta_t))
                tf.summary.scalar('annealing beta', annealing)
        else:
            annealing = 1.

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))
            tf.summary.scalar('Loglike', cross_entropy)

        with tf.name_scope('lower_bound'):
            lowerbound = cross_entropy + annealing * regs
            tf.summary.scalar('Lower bound', lowerbound)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('Accuracy/Train', accuracy)
            tf.summary.scalar('Accuracy/Test', test_accuracy)

    tf.add_to_collection('logits', logits)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('training', training)
    tf.add_to_collection('sample', sample)
    tf.add_to_collection('use_theta', use_theta)
    tf.add_to_collection('x', input_x)
    tf.add_to_collection('y', input_y)

    if FLAGS.sparsity != 'none':
        for i, layer in enumerate(layers):
            log_a, log_amat = layer.get_loga_sparsity()
            tf.add_to_collection('loga_' + str(i), log_a)
            tf.add_to_collection('logamat_' + str(i), log_amat)
            w, b = layer.sample_theta()
            tf.add_to_collection('theta_W_' + str(i), w)
            tf.add_to_collection('theta_b_' + str(i), b)

    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    starter_learning_rate = FLAGS.lr
    num_steps = iter_per_epoch * FLAGS.epochs
    learning_rate = tf.minimum(starter_learning_rate,
                               2 * starter_learning_rate * tf.maximum(0., (num_steps - tf.cast(global_step, tf.float32)) / num_steps))
    tf.summary.scalar('learning_rate', learning_rate)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lowerbound, global_step=global_step)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    saver = tf.train.Saver(tf.global_variables())
    tf.global_variables_initializer().run()

    merged = tf.summary.merge_all()
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    steps = 0
    model_dir = './models/vgglike_cifar10_{}_da{}_bn{}_anneal{}_k{}_l2{}/model/'.format(FLAGS.sparsity, FLAGS.da, FLAGS.bn,
                                                                                 FLAGS.anneal, FLAGS.model_size, FLAGS.l2)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Train
    for epoch in xrange(FLAGS.epochs):
        widgets = ["epoch {}/{}|".format(epoch + 1, FLAGS.epochs), Percentage(), Bar(), ETA()]
        pbar = ProgressBar(iter_per_epoch, widgets=widgets)
        pbar.start()
        t0 = time.time()

        gen = batch_iterator(xtrain, ytrain, 100, data_augmentation=FLAGS.da)
        for j in xrange(iter_per_epoch):
            steps += 1
            pbar.update(j)
            batch_x, batch_y = next(gen)
            summary, _, _ = sess.run([merged, train_step, extra_update_ops], feed_dict={input_x: batch_x, input_y: batch_y,
                                                                   sample: True, training: True, use_theta: False})
            train_writer.add_summary(summary, steps)
            train_writer.flush()

        string = 'Epoch {}/{}'.format(epoch + 1, FLAGS.epochs)

        accs, naccs = 0., 0.
        for b in xrange(ytest.shape[0] / 100):
            tacc = sess.run(accuracy,
                            feed_dict={input_x: xtest[b * 100:(b + 1) * 100], input_y: ytest[b * 100:(b + 1) * 100],
                                       training: False, sample: False, use_theta: False})
            naccs += 1
            accs += tacc
        tacc = accs / naccs
        string += 'test_acc: {:0.3f}'.format(tacc)
        sess.run(assign_test_acc, feed_dict={test_acc_placeholder: tacc})

        if (epoch + 1) % FLAGS.save_every == 0:
            string += ', model_save: True'
            saver.save(sess, model_dir + 'model')

        string += ', dt: {:0.3f}'.format(time.time() - t0)
        print string

    saver.save(sess, model_dir + 'model')
    pyx = tf.nn.softmax(tf.get_collection('logits')[0])

    preds = np.zeros_like(ytest)
    widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.L, widgets=widgets)
    pbar.start()
    for i in xrange(FLAGS.L):
        pbar.update(i)
        for j in xrange(xtest.shape[0] / 100):
            pyxi = sess.run(pyx, feed_dict={input_x: xtest[j * 100:(j + 1) * 100], training: False, sample: True,
                                            use_theta: False})
            preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
    print
    sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(ytest, 1)))
    print 'Sample accuracy: {}'.format(sample_accuracy)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='',
                        help='Summaries directory')
    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-epzero', type=int, default=1)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-L', type=int, default=100)
    parser.add_argument('-anneal', action='store_true')
    parser.add_argument('-sparsity', choices=['none', 'group', 'weight', 'hs++'], default='none')
    parser.add_argument('-model_size', default=1, type=float)
    parser.add_argument('-save_every', default=50, type=int)
    parser.add_argument('-fq', default=0, type=int)
    parser.add_argument('-share', action='store_true')
    parser.add_argument('-da', action='store_true', help='Data augmentation')
    parser.add_argument('-sparsity_rate', type=float, default=0.1)
    parser.add_argument('-t0', type=float, default=0.01)
    parser.add_argument('-logging', action='store_true')
    parser.add_argument('-sa', type=float, default=0.1)
    parser.add_argument('-sb', type=float, default=0.1)
    parser.add_argument('-l2', type=float, default=0., help='L2 regularization of weight matrices')
    parser.add_argument('-bn', action='store_true')
    parser.add_argument('-pretrained_model', type=str, default=None)
    FLAGS = parser.parse_args()
    if FLAGS.summaries_dir == '':
        FLAGS.summaries_dir = './logs/vgglike_cifar10_{}_da{}_bn{}_anneal{}_k{}_l2{}'.format(FLAGS.sparsity, FLAGS.da,
                                                                                             FLAGS.bn, FLAGS.anneal,
                                                                                             FLAGS.model_size, FLAGS.l2)
        FLAGS.summaries_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    main()
