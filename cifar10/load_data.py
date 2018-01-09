import numpy as np
import os
import cPickle as pickle
import random
from sklearn.preprocessing import OneHotEncoder

def unpickle(fname):
    import cPickle
    with open(fname, 'rb') as fo:
        d = cPickle.load(fo)
    return d


def load(dataset):
    if dataset == 'mnist':
        return load_mnist()
    if dataset == 'cifar10':
        return load_cifar10()
    if dataset == 'cifar10-random':
        return load_cifar10_random()
    if dataset == 'cifar100':
        return load_cifar100()
    if dataset == 'imagenet':
        return load_imagenet()


def load_pure(dataset):
    if dataset == 'cifar10':
        return load_CIFAR10('/home/ashuha/atanov/data/cifar-10-batches-py')


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in xrange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def iterate_minibatches_trio(inputs, targets, imgs, batchsize, shuffle=False):
    assert len(inputs) == len(targets) and len(inputs) == len(imgs)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in xrange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], imgs[excerpt]


def batch_iterator_train_crop_flip(data, y, batchsize, shuffle=False):
    PIXELS = 32
    PAD_CROP = 4
    n_samples = data.shape[0]
    # Shuffles indicies of training data, so we can draw batches from random indicies instead of shuffling whole data
    indx = np.random.permutation(xrange(n_samples))
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[indx[sl]]
        y_batch = y[indx[sl]]

        # pad and crop settings
        trans_1 = random.randint(0, (PAD_CROP*2))
        trans_2 = random.randint(0, (PAD_CROP*2))
        crop_x1 = trans_1
        crop_x2 = (PIXELS + trans_1)
        crop_y1 = trans_2
        crop_y2 = (PIXELS + trans_2)

        # flip left-right choice
        flip_lr = random.randint(0,1)

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.copy(X_batch)

        # for each image in the batch do the augmentation
        for j in range(X_batch.shape[0]):
            # for each image channel
            for k in range(X_batch.shape[-1]):
                # pad and crop images
                img_pad = np.pad(
                    X_batch_aug[j,:,:, k], pad_width=((PAD_CROP, PAD_CROP), (PAD_CROP, PAD_CROP)), mode='constant')
#                 print(X_batch_aug[j,:,:, k].shape, img_pad.shape)
                X_batch_aug[j,:,:,k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                # flip left-right if chosen
                if flip_lr == 1:
                    X_batch_aug[j,:,:,k] = np.fliplr(X_batch_aug[j,:,:,k])

        # fit model on each batch
        yield X_batch_aug, y_batch, indx[sl]


def load_CIFAR10(ROOT):
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            Y = np.array(datadict['labels'])
            X = datadict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            return X / 255., Y

    xs, ys = [], []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T, x) / x.shape[0]
        U, S, V = np.linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1. / np.sqrt(S + self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + self.regularization)))
        self.ZCA_mat = np.dot(tmp, U.T)
        self.inv_ZCA_mat = np.dot(tmp2, U.T)
        self.mean = m.copy()

    def apply(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return np.dot(x.reshape((s[0], np.prod(s[1:]))) - self.mean, self.ZCA_mat).reshape(
                s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")

    def invert(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return (
            np.dot(x.reshape((s[0], np.prod(s[1:]))), self.inv_ZCA_mat.get_value()) + self.mean.get_value()).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")



def load_cifar10(base='/home/ashuha/atanov/data/', normalize=True, add_noise=False):
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            Y = np.array(datadict['labels'])
            X = datadict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            return X / 255., Y

    def load_CIFAR10(ROOT):
        xs, ys = [], []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(base, 'cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Normalize the data: subtract the mean image

    if add_noise and normalize:
        print('!!!! ADD noise in data and normalize it both turn on !!!!')

    if normalize:
        whitener = ZCA(x=X_train)
        X_train = whitener.apply(X_train)
        X_test = whitener.apply(X_test)

    # Transpose so that channels come first
    # X_train = X_train.transpose(0, 3, 1, 2).copy()
    # X_test = X_test.transpose(0, 3, 1, 2).copy()

    enc = OneHotEncoder(sparse=False)
    enc.fit(y_train[:,np.newaxis])
    y_train = enc.transform(y_train[:,np.newaxis])
    y_test = enc.transform(y_test[:,np.newaxis])

    if add_noise:
        X_noise = np.load(base + '/cifar-10-noise-imgs.npy')
        y_noise = np.load(base + '/cifar-10-noise-labels.npy')
        X_train = np.concatenate((X_train, X_noise), axis=0)
        y_train = np.concatenate((y_train, y_noise), axis=0)

    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 32, 32, 3), 10
