"""Script to test a trained model."""
import os
import sys
import getopt

import numpy as np
import models as m
import matplotlib.pyplot as plt
import util.util as u

from util.data import TwoImageIterator
from util.util import MyDict, load_params, load_weights_of, compose_imgs, convert_to_rgb, mkdir, get_log_dir


def print_help():
    """Print how to use this script."""
    print "Usage:"
    print "test.py [--help] [--results_dir] [--log_dir] [--base_dir] [--train_dir] [--val_dir] " \
          "[--test_dir] [--load_to_memory] [--expt_name] [--target_size] [--N]"
    print "--results_dir: Directory where to save the results."
    print "--log_dir': Directory where the experiment was logged."
    print "--base_dir: Directory that contains the data."
    print "--train_dir: Directory inside base_dir that contains training data."
    print "--val_dir: Directory inside base_dir that contains validation data."
    print "--test_dir: Directory inside base_dir that contains test data."
    print "--load_to_memory: Whether to load the images into memory."
    print "--expt_name: The name of the experiment to test."
    print "--target_size: The size of the images loaded by the iterator."
    print "--N: The number of samples to generate."


def join_and_create_dir(*paths):
    """Join the paths provided as arguments, create the directory and return the path."""
    path = os.path.join(*paths)
    mkdir(path)

    return path


def save_pix2pix(unet, it, path, params):
    """Save the results of the pix2pix model."""
    real_dir = join_and_create_dir(path, 'real')
    a_dir = join_and_create_dir(path, 'A')
    b_dir = join_and_create_dir(path, 'B')
    comp_dir = join_and_create_dir(path, 'composed')

    for i, filename in enumerate(it.filenames):
        a, b = next(it)
        bp = unet.predict(a)
        bp = convert_to_rgb(bp[0], is_binary=params.is_b_binary)

        img = compose_imgs(a[0], b[0], is_a_binary=params.is_a_binary, is_b_binary=params.is_b_binary)
        hi, wi, chi = img.shape
        hb, wb, chb = bp.shape
        if hi != hb or wi != 2*wb or chi != chb:
            raise Exception("Mismatch in img and bp dimensions {0} / {1}".format(img.shape, bp.shape))

        composed = np.zeros((hi, wi+wb, chi))
        composed[:, :wi, :] = img
        composed[:, wi:, :] = bp

        a = convert_to_rgb(a[0], is_binary=params.is_a_binary)
        b = convert_to_rgb(b[0], is_binary=params.is_b_binary)

        plt.imsave(open(os.path.join(real_dir, filename), 'wb+'), b)
        plt.imsave(open(os.path.join(b_dir, filename), 'wb+'), bp)
        plt.imsave(open(os.path.join(a_dir, filename), 'wb+'), a)
        plt.imsave(open(os.path.join(comp_dir, filename), 'wb+'), composed)


def save_all_pix2pix(unet, it_train, it_val, it_test, params):
    """Save all the results of the pix2pix model."""
    expt_dir = get_log_dir(params.results_dir, params.expt_name)

    # Create directores if they do not exist
    mkdir(params.results_dir)
    mkdir(expt_dir)

    train_dir = join_and_create_dir(expt_dir, params.train_dir)
    val_dir = join_and_create_dir(expt_dir, params.val_dir)
    test_dir = join_and_create_dir(expt_dir, params.test_dir)

    save_pix2pix(unet, it_train, train_dir, params)
    save_pix2pix(unet, it_val, val_dir, params)
    save_pix2pix(unet, it_test, test_dir, params)


if __name__ == '__main__':
    a = sys.argv[1:]

    params = MyDict({
        'results_dir': 'results',  # Directory where to save the results
        'log_dir': 'log',  # Directory where the experiment was logged
        'base_dir': 'data/unet_segmentations_binary',  # Directory that contains the data
        'train_dir': 'train',  # Directory inside base_dir that contains training data
        'val_dir': 'val',  # Directory inside base_dir that contains validation data
        'test_dir': 'test',  # Directory inside base_dir that contains test data
        'load_to_memory': True,  # Whether to load the images into memory
        'expt_name': None,  # The name of the experiment to test
        'target_size': 512,  # The size of the images loaded by the iterator
        'N': 100,  # The number of samples to generate
    })

    param_names = [k + '=' for k in params.keys()] + ['help']

    try:
        opts, args = getopt.getopt(a, '', param_names)
    except getopt.GetoptError:
        print_help()
        sys.exit()

    for opt, arg in opts:
        if opt == '--help':
            print_help()
            sys.exit()
        elif opt in ('--target_size', '--N'):
            params[opt[2:]] = int(arg)
        elif opt in ('--load_to_memory'):
            params[opt[2:]] = True if arg == 'True' else False
        elif opt in ('--results_dir', '--log_dir', '--base_dir', '--train_dir',
                     '--val_dir', '--test_dir', '--expt_name'):
            params[opt[2:]] = arg

    params = load_params(params)
    params = MyDict(params)

    # Define the U-Net generator
    unet = m.g_unet(params.a_ch, params.b_ch, params.nfatob, is_binary=params.is_b_binary)
    load_weights_of(unet, u.ATOB_WEIGHTS_FILE, log_dir=params.log_dir, expt_name=params.expt_name)

    ts = params.target_size
    train_dir = os.path.join(params.base_dir, params.train_dir)
    it_train = TwoImageIterator(train_dir,  is_a_binary=params.is_a_binary,
                                is_a_grayscale=params.is_a_grayscale,
                                is_b_grayscale=params.is_b_grayscale,
                                is_b_binary=params.is_b_binary, batch_size=1,
                                load_to_memory=params.load_to_memory,
                                target_size=(ts, ts), shuffle=False)
    val_dir = os.path.join(params.base_dir, params.val_dir)
    it_val = TwoImageIterator(val_dir,  is_a_binary=params.is_a_binary,
                              is_b_binary=params.is_b_binary,
                              is_a_grayscale=params.is_a_grayscale,
                              is_b_grayscale=params.is_b_grayscale, batch_size=1,
                              load_to_memory=params.load_to_memory,
                              target_size=(ts, ts), shuffle=False)
    test_dir = os.path.join(params.base_dir, params.test_dir)
    it_test = TwoImageIterator(test_dir,  is_a_binary=params.is_a_binary,
                               is_b_binary=params.is_b_binary,
                               is_a_grayscale=params.is_a_grayscale,
                               is_b_grayscale=params.is_b_grayscale, batch_size=1,
                               load_to_memory=params.load_to_memory,
                               target_size=(ts, ts), shuffle=False)

    save_all_pix2pix(unet, it_train, it_val, it_test, params)
