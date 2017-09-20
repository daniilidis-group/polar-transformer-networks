# This is part of the demo source code for the paper:
# Esteves, C., Allen-Blanchette, C., Zhou, X. and Daniilidis, K., 2017. Polar Transformer Networks. arXiv preprint arXiv:1709.01889.  http://arxiv.org/abs/1709.01889v1
# GRASP Laboratory - University of Pennsylvania
# http://github.com/daniilidis-group/polar-transformer-networks


import os

import numpy as np

import tensorflow as tf

from tflearn.layers.core import input_data
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, global_avg_pool
import tflearn

import layers


def get_input_layer(flags):
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    curr = input_data(shape=[None, flags.inres, flags.inres, flags.inchan],
                      name='input',
                      data_preprocessing=img_prep)

    if flags.rts_aug:
        w, h = curr.get_shape().as_list()[1:3]
        a = -flags.rts_aug_ang + 2*flags.rts_aug_ang*tf.random_uniform([flags.bs])
        a *= np.pi/180
        # centralize rot/scale
        y = ((w - 1) - (tf.cos(a)*(w-1) - tf.sin(a)*(h-1))) / 2.0
        x = ((h - 1) - (tf.sin(a)*(w-1) + tf.cos(a)*(h-1))) / 2.0
        transforms = tf.transpose(tf.stack([tf.cos(a), tf.sin(a), x,
                                            -tf.sin(a), tf.cos(a), y,
                                            tf.zeros(flags.bs), tf.zeros(flags.bs)]))
        
        return tf.cond(tflearn.get_training_mode(),
                       lambda: tf.contrib.image.transform(curr, transforms),
                       lambda: curr)
    else:
        return curr


def finalize_get_model(net, flags):
    net['gap'], curr = dup(global_avg_pool(net['conv_final'], name='gap'))

    net['final'] = regression(curr,
                              optimizer='adam',
                              learning_rate=flags.lr,
                              batch_size=flags.bs,
                              loss='softmax_categorical_crossentropy',
                              name='target',
                              n_classes=flags.nc,
                              shuffle_batches=True)

    model = tflearn.DNN(net['final'],
                        tensorboard_verbose=0,
                        tensorboard_dir=flags.logdir,
                        best_checkpoint_path=os.path.join(flags.logdir,
                                                          flags.run_id,
                                                          flags.run_id),
                        best_val_accuracy=flags.acc_save)

    model.net_dict = net
    model.flags = flags

    return model


def polar_transformer_network_layers(flags):
    net = {}

    net['input'] = get_input_layer(flags)

    regr = pt_regressor(net['input'], flags)

    for k, v in regr.items():
        net[k] = v

    log = True if flags.polarmode == 'log' else False
    net['polar'], curr = dup(layers.polar_transformer(net['input'],
                                                      net['polar_origin'],
                                                      (flags.inres, flags.inres),
                                                      log=log,
                                                      radius_factor=flags.polar_rf)[..., 0][..., np.newaxis])

    # classifier network
    cl = conv_from_flags(curr, flags)
    for k, v in cl.items():
        net[k] = v

    return net


def polar_transformer_network(flags):
    net = polar_transformer_network_layers(flags)
    model = finalize_get_model(net, flags)
    return model


def pt_regressor(layer_in, flags):
    net, curr = pt_regressor_conv(layer_in, flags)

    net['ptreg_in'] = layer_in

    dims = curr.get_shape().as_list()
    weights_init = 'zeros'
    bias_init = tf.ones([1])

    # 1x1 conv, no BN, no ReLU on final heatmap
    net['ptreg_out'], curr = dup(conv_2d(curr, 1, 1, activation='linear',
                                         weights_init=weights_init,
                                         bias_init=bias_init,
                                         padding=flags.pad,
                                         name='ptreg_out'))
    # take the centroid of the feature map
    s = tf.shape(curr)
    # compute xc, yc from -1 to 1
    xc = tf.tile(tf.linspace(-1., 1., s[2])[np.newaxis, ...],
                 (s[1], 1))
    yc = tf.transpose(xc)

    net['po_j'] = (tf.reduce_sum(curr[..., 0]*xc[np.newaxis, ...], axis=(1, 2)) /
                   tf.reduce_sum(curr[..., 0], axis=(1, 2)))
    net['po_i'] = (tf.reduce_sum(curr[..., 0]*yc[np.newaxis, ...], axis=(1, 2)) /
                   tf.reduce_sum(curr[..., 0], axis=(1, 2)))
    net['polar_origin'] = tf.stack([net['po_j'], net['po_i']], axis=1)

    # origin augmentation
    if flags.ptreg_aug > 0:
        dim = layer_in.get_shape().as_list()[1]
        shift = tf.cond(tflearn.get_training_mode(),
                        lambda: 1./dim * tf.random_uniform([flags.bs, 2],
                                                  minval=-flags.ptreg_aug,
                                                  maxval=flags.ptreg_aug),
                        lambda: tf.zeros([flags.bs, 2]))
        net['polar_origin'] += shift

    return net


def pt_regressor_conv(layer_in, flags):
    """ Return standard convolutional polar transform origin regressor """
    nfilters = [int(x) for x in flags.ptreg_nfilters.split(",")]
    strides = [int(x) for x in flags.ptreg_strides.split(",")]
    weights_init = flags.weights_init

    # first block is always conv_bn_relu
    name1st = 'ptreg_conv0'
    first = layers.conv_bn_relu(layer_in, nfilters[0], 3, name1st,
                                strides=strides[0],
                                padding=flags.pad,
                                weight_decay=flags.weight_decay,
                                weights_init=weights_init,
                                activation=flags.activation)
    nfilters = nfilters[1:]
    strides = strides[1:]

    net, curr = layers.conv_sequence(first,
                                     nfilters,
                                     strides,
                                     block_fun=layers.conv_bn_relu,
                                     pad=flags.pad,
                                     weight_decay=flags.weight_decay,
                                     weights_init=weights_init,
                                     name_prefix='ptreg_',
                                     activation=flags.activation)

    net[name1st] = first
    return net, curr


def conv_from_flags(layer_in, flags):
    """ Build series of strided convolutional layers from flags.nfilters and flags.strides. """
    nfilters = [int(x) for x in flags.nfilters.split(",")]
    strides = [int(x) for x in flags.strides.split(",")]
    block_fun = layers.conv_bn_relu
    net = {}

    # first block is always conv_bn_relu
    if flags.pad_wrap:
        pad = 'wrap'
    else:
        pad = flags.pad
    net['conv0'], curr = dup(layers.conv_bn_relu(layer_in, nfilters[0], flags.filter_size, 'conv0',
                                                 strides=strides[0],
                                                 padding=pad,
                                                 weight_decay=flags.weight_decay,
                                                 weights_init=flags.weights_init,
                                                 activation=flags.activation))
    nfilters = nfilters[1:]
    strides = strides[1:]

    seq, curr = layers.conv_sequence(curr, nfilters, strides, block_fun,
                                     pad=pad,
                                     weight_decay=flags.weight_decay,
                                     activation=flags.activation,
                                     filter_size=flags.filter_size)

    for k, v in seq.items():
        net[k] = v

    net = finalize_conv_from_flags(net, curr, flags)

    return net


def finalize_conv_from_flags(net, curr, flags):
    if flags.pad_wrap:
        curr = layers.wrap_pad_rows(curr)
        pad = 'valid'

    # final layer is linear
    name = 'conv_final'
    net[name] = conv_2d(curr, flags.nc, flags.filter_size,
                        activation='linear', name=name, padding=pad)

    return net


def dup(x):
    """ Return two references for input; useful when creating NNs and storing references to layers """
    return [x, x]
