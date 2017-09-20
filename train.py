# This is part of the demo source code for the paper:
# Esteves, C., Allen-Blanchette, C., Zhou, X. and Daniilidis, K., 2017. Polar Transformer Networks. arXiv preprint arXiv:1709.01889.  http://arxiv.org/abs/1709.01889v1
# GRASP Laboratory - University of Pennsylvania
# http://github.com/daniilidis-group/polar-transformer-networks


import os
import shutil
import pickle
import time

import numpy as np
import tensorflow as tf

import arch
import flags_handler
import util


def main(argv):
    flags = flags_handler.expand_flags(tf.app.flags.FLAGS)
    flags_handler.check_flags(flags, argv)

    X, Y, valX, valY, testX, testY = util.train_test_val_mnist(os.path.expanduser(flags.datadir))

    if flags.combine_train_val:
        X = np.concatenate([X, valX])
        Y = np.concatenate([Y, valY])
        val = (X, Y)
    else:
        val = (valX, valY)

    # this is picked up by tflearn trainer when model.DNN is instantiated
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.add_to_collection(tf.GraphKeys.GRAPH_CONFIG, config)

    model = getattr(arch, flags.arch)(flags)

    # traverse directory and compute run_id
    if flags.run_id == "":
        for i in range(1000):
            flags.run_id = run_id = "run{:03}".format(i)
            dirname = "{}/{}".format(model.trainer.tensorboard_dir, run_id)
            if not os.path.isdir(dirname):
                break
    else:
        run_id = flags.run_id
        dirname = "{}/{}".format(model.trainer.tensorboard_dir, run_id)
        if os.path.isdir(dirname):
            print('Warning: log directory not empty. moving to {}.bkp'.format(dirname))
            shutil.rmtree(dirname + '.bkp', ignore_errors=True)
            shutil.move(dirname, dirname + '.bkp')

    print('model dir={}'.format(dirname))

    # save flags
    os.makedirs(dirname)
    flagsfile = '{}/{}'.format(dirname, 'flags.pickle')
    print('Saving flags to {}'.format(flagsfile))
    with open(flagsfile, 'wb') as fout:
        pickle.dump(flags.__flags, fout)

    t0 = time.time()
    fit = getattr(model, 'custom_fit', model.fit)
    fit(X, Y,
        n_epoch=flags.ne,
        validation_set=val,
        snapshot_epoch=True,
        show_metric=True,
        run_id=run_id)
    traintime = time.time() - t0

    # save final set of weights
    model.save(os.path.join(flags.logdir, run_id, 'final'))

    # load best model
    basename = os.path.join(flags.logdir, run_id, run_id)
    best_model = util.best_model_from_dir(basename)
    if best_model:
        print('loading best model from ' + best_model)
        model.load(best_model)
    else:
        print('Warning! Could not find best model; using final...')

    evaluate = getattr(model, 'custom_evaluate', model.evaluate)
    print('Evaluating best model on test set (no augmentation)...')
    test = evaluate(testX, testY, batch_size=flags.bs)[0]

    print('Evaluating best model on augmented test set...')
    combined = util.predict_testaug(model, testX, angs=np.arange(0, 360, 45))
    final_pred = np.argmax(combined, axis=1)
    testaug = sum(final_pred == np.argmax(testY, axis=1))/len(testY)
    
    # test = evaluate(testX, testY, batch_size=flags.bs)[0]
    aug = 'train augmentation.' if flags.rts_aug else ''
    print('{}. {} # of params: {}. Training time: {:.2f} s.'.
          format(run_id, aug, util.count_weights(print_perlayer=False), traintime))
    print('Test accuracy (no test time augmentation): {:.4f}'.format(test))
    print('Test accuracy (with test time augmentation): {:.4f}'.format(testaug))


if __name__ == '__main__':
    flags_handler.define_flags()
    tf.app.run()
