# This is part of the demo source code for the paper:
# Esteves, C., Allen-Blanchette, C., Zhou, X. and Daniilidis, K., 2017. Polar Transformer Networks. arXiv preprint arXiv:1709.01889.  http://arxiv.org/abs/1709.01889v1
# GRASP Laboratory - University of Pennsylvania
# http://github.com/daniilidis-group/polar-transformer-networks


import glob
import re
import itertools
from joblib import Parallel, delayed
import subprocess
import os

import numpy as np
from skimage.transform import rotate
import tensorflow as tf


def to_one_hot(v):
    """ Convert vector to one hot form. """
    n = len(v)
    m = max(v) + 1
    out = np.zeros((n, m))
    out[np.arange(n), v] = 1
    return out


def softmax(x, axis=0):
    assert axis in [0, 1]

    min_ax = x.min(axis=axis)
    min_ax = min_ax[:, np.newaxis] if axis == 1 else min_ax
    den = np.exp(x - min_ax).sum(axis=axis)
    den = den[:, np.newaxis] if axis == 1 else den

    return np.exp(x - min_ax) / den


def grouper(iterable, n, fillvalue=None):
    """ Iterate over chunks of iterable.

    Note: will fill last value with None if size is not a multiple of n.

    From: http://stackoverflow.com/a/434411/6079076
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def train_test_val_mnist(datadir):
    mnist_dir = datadir + '/mnist_rotation_new'

    # download model if it doesn't exist
    fnames = ['rotated_train.npz', 'rotated_valid.npz', 'rotated_test.npz']
    try:
        train, valid, test = [np.load(os.path.join(mnist_dir, f)) for f in fnames]
    except:
        print('Dataset not found at {}. Downloading it'.format(mnist_dir))
        os.makedirs(mnist_dir, exist_ok=True)
        # note: mnist_rotation_new.zip is the same file used in the Harmonic Networks: https://github.com/deworrall92/harmonicConvolutions
        subprocess.call(['wget',
                         '--no-check-certificate',
                         'http://seas.upenn.edu/~machc/data/mnist_rotation_new.zip',
                         '-O', os.path.join(mnist_dir, 'tmp.zip')])
        subprocess.call(['unzip',
                         os.path.join(mnist_dir, 'tmp.zip'),
                         '-d', datadir])

        train, valid, test = [np.load(os.path.join(mnist_dir, f)) for f in fnames]

    X = train['x'].reshape([-1, 28, 28, 1])
    Y = to_one_hot(train['y'])
    valX = valid['x'].reshape([-1, 28, 28, 1])
    valY = to_one_hot(valid['y'])
    testX = test['x'].reshape([-1, 28, 28, 1])
    testY = to_one_hot(test['y'])

    return X, Y, valX, valY, testX, testY


def best_model_from_dir(basename):
    """ Return best saved model from basename. """
    models = glob.glob(basename + '*.index')
    best_model = None
    # get best model, if exists
    models_out = []
    for m in models:
        match = re.match(re.escape(basename) + '(1?[0-9]{4}).index', m)
        if match:
            models_out.append(int(match.groups()[0]))

    if models_out:
        acc = max(models_out)
        best_model = basename + str(acc)

    return best_model


def count_weights(print_perlayer=True):
    """ Count number of trainable variables on current tf graph. """
    acc_total = 0
    for v in tf.trainable_variables():
        dims = v.get_shape().as_list()
        total = np.prod(dims)
        acc_total += total
        if print_perlayer:
            print('{}: {}, {}'.format(v.name, dims, total))

    return acc_total


def predict_batches(model, X, batchsize=None):
    """ run tflearn.DNN.predict() in batches for a model. """
    if batchsize is None:
        batchsize = model.flags.bs
    pred = []
    for batch in grouper(X, batchsize):
        pred.append(model.predict(np.array(batch)))

    return np.concatenate(pred)


def predict_testaug(model, X, batchsize=None, angs=None):
    """ Run predictions w/ test time rotation augmentation by angs"""
    preds = []
    for a in angs:
        print('rotating test set by angle: {:.2f}...'.format(a))
        rotX = np.stack(Parallel(n_jobs=-1)(delayed(rotate)
                                            (im, a, preserve_range=True)
                                            for im in X))
        preds.append(predict_batches(model, rotX, batchsize=batchsize))
    combined = sum([softmax(p, axis=1) for p in preds])

    return combined
