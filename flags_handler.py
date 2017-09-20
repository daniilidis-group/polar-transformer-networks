# This is part of the demo source code for the paper:
# Esteves, C., Allen-Blanchette, C., Zhou, X. and Daniilidis, K., 2017. Polar Transformer Networks. arXiv preprint arXiv:1709.01889.  http://arxiv.org/abs/1709.01889v1
# GRASP Laboratory - University of Pennsylvania
# http://github.com/daniilidis-group/polar-transformer-networks


import os

import tensorflow as tf


def define_flags():
    # architecture
    tf.app.flags.DEFINE_string("arch", "ptn", "Base architecture to train")
    tf.app.flags.DEFINE_string("nfilters", "big28", "Number of filters per conv layer")
    tf.app.flags.DEFINE_integer("filter_size", 3, "Size of convolutional kernel")
    tf.app.flags.DEFINE_string("strides", "big28", "Strides on each conv per layer")
    tf.app.flags.DEFINE_string("pad", "same", "Padding mode")
    tf.app.flags.DEFINE_string("activation", "relu", "Default nonlinearity")

    # polar transformer options
    tf.app.flags.DEFINE_string("polarmode", "log", "log or linear")
    tf.app.flags.DEFINE_float("polar_rf", 0.7071, "polar transform max radius (factor of width)")
    tf.app.flags.DEFINE_string("ptreg_nfilters", "20,20,20", "pt regressor number of filters per conv")
    tf.app.flags.DEFINE_string("ptreg_strides", "2,1,1", "pt regressor strides per conv")
    tf.app.flags.DEFINE_float("ptreg_aug", 4, "polar origin augmentation. Value is maximum shift. ")
    tf.app.flags.DEFINE_bool("pad_wrap", True, "Use row wrap padding mode in classifier network")

    # dataset
    tf.app.flags.DEFINE_string("datadir", "/tmp", "dataset directory")
    tf.app.flags.DEFINE_integer("nc", 10, "Number of classes")
    tf.app.flags.DEFINE_integer("inres", 28, "Resolution of input")
    tf.app.flags.DEFINE_integer("inchan", 1, "Number of channels of the input")
    tf.app.flags.DEFINE_bool("standardize", True, "Standardize input (subtract mean, divide by std)")
    tf.app.flags.DEFINE_bool("mean_sub", False, "Subtract mean in preprocessing step")
    tf.app.flags.DEFINE_bool("combine_train_val", True, "Use combination of train and val set for training.")

    # training
    tf.app.flags.DEFINE_integer("ne", 500, "Number of epochs to train")
    tf.app.flags.DEFINE_float("lr", 0.01, "Learning rate")
    tf.app.flags.DEFINE_float("weight_decay", 0., "regularization weight decay")
    tf.app.flags.DEFINE_string('weights_init', 'variance_scaling', 'weight initialization scheme')
    tf.app.flags.DEFINE_integer("bs", 200, "Batch size")
    tf.app.flags.DEFINE_bool("rts_aug", True, "Augment input with random rotations, translations, scaling")
    tf.app.flags.DEFINE_bool("rts_aug_ang", 180, "rts_aug +- angles")
    tf.app.flags.DEFINE_string("logdir", os.path.expanduser("/tmp"), "log directory")
    tf.app.flags.DEFINE_string("run_id", "", "log subdirectory; will be runxxx if empty")
    tf.app.flags.DEFINE_float("acc_save", 0.95, "Min accuracy to save best model")


def expand_flags(flags):
    aliases = {'arch': {'ptn': 'polar_transformer_network'},
               'nfilters': {'small28': '20,20,20,20,20,20',
                            'big28': '16,16,32,32,32,64,64,64',
                            'small42': '20,20,20,20,20,20,20',
                            'big42': '16,16,16,16,32,32,32,64,64,64',
                            'small96': '20,20,20,20,20,20,20,20',
                            'big96': '16,16,16,16,32,32,32,64,64,64'},
               'strides': {'small28': '1,2,1,1,1,1',
                           'big28': '1,1,2,1,1,2,1,1',
                           'small42': '2,1,2,1,1,1,1',
                           'big42': '1,2,1,1,2,1,1,2,1,1',
                           'small96': '2,2,1,2,1,1,1,1',
                           'big96': '2,2,1,1,2,1,1,2,1,1'}}

    for k, v in aliases.items():
        for kk, vv in v.items():
            if getattr(flags, k) == kk:
                setattr(flags, k, vv)

    return flags


def check_flags(flags, argv=None):
    """ Sanity check on flags """
    assert len(argv) == 1, 'Unrecognized flags: {}'.format(' '.join(argv[1:]))
    assert flags.pad in ['same', 'valid']
    assert flags.polarmode in ['log', 'linear']
