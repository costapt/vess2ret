"""Auxiliary methods."""
import os
import json
from errno import EEXIST

import numpy as np
import seaborn as sns
import cPickle as pickle
import matplotlib.pyplot as plt

sns.set()

DEFAULT_LOG_DIR = 'log'
ATOB_WEIGHTS_FILE = 'atob_weights.h5'
D_WEIGHTS_FILE = 'd_weights.h5'


class MyDict(dict):
    """
    Dictionary that allows to access elements with dot notation.

    ex:
        >> d = MyDict({'key': 'val'})
        >> d.key
        'val'
        >> d.key2 = 'val2'
        >> d
        {'key2': 'val2', 'key': 'val'}
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def convert_to_rgb(img, is_binary=False):
    """Given an image, make sure it has 3 channels and that it is between 0 and 1."""
    if len(img.shape) != 3:
        raise Exception("""Image must have 3 dimensions (channels x height x width). """
                        """Given {0}""".format(len(img.shape)))

    img_ch, _, _ = img.shape
    if img_ch != 3 and img_ch != 1:
        raise Exception("""Unsupported number of channels. """
                        """Must be 1 or 3, given {0}.""".format(img_ch))

    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)

    if not is_binary:
        imgp = imgp * 127.5 + 127.5
        imgp /= 255.

    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)


def compose_imgs(a, b, is_a_binary=True, is_b_binary=False):
    """Place a and b side by side to be plotted."""
    ap = convert_to_rgb(a, is_binary=is_a_binary)
    bp = convert_to_rgb(b, is_binary=is_b_binary)

    if ap.shape != bp.shape:
        raise Exception("""A and B must have the same size. """
                        """{0} != {1}""".format(ap.shape, bp.shape))

    # ap.shape and bp.shape must have the same size here
    h, w, ch = ap.shape
    composed = np.zeros((h, 2*w, ch))
    composed[:, :w, :] = ap
    composed[:, w:, :] = bp

    return composed


def get_log_dir(log_dir, expt_name):
    """Compose the log_dir with the experiment name."""
    if log_dir is None:
        raise Exception('log_dir can not be None.')

    if expt_name is not None:
        return os.path.join(log_dir, expt_name)
    return log_dir


def mkdir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise


def create_expt_dir(params):
    """Create the experiment directory and return it."""
    expt_dir = get_log_dir(params.log_dir, params.expt_name)

    # Create directories if they do not exist
    mkdir(params.log_dir)
    mkdir(expt_dir)

    # Save the parameters
    json.dump(params, open(os.path.join(expt_dir, 'params.json'), 'wb'),
              indent=4, sort_keys=True)

    return expt_dir


def plot_loss(loss, label, filename, log_dir):
    """Plot a loss function and save it in a file."""
    plt.figure(figsize=(5, 4))
    plt.plot(loss, label=label)
    plt.legend()
    plt.savefig(os.path.join(log_dir, filename))
    plt.clf()


def log(losses, atob, it_val, N=4, log_dir=DEFAULT_LOG_DIR, expt_name=None,
        is_a_binary=True, is_b_binary=False):
    """Log losses and atob results."""
    log_dir = get_log_dir(log_dir, expt_name)

    # Save the losses for further inspection
    pickle.dump(losses, open(os.path.join(log_dir, 'losses.pkl'), 'wb'))

    ###########################################################################
    #                             PLOT THE LOSSES                             #
    ###########################################################################
    plot_loss(losses['d'], 'discriminator', 'd_loss.png', log_dir)
    plot_loss(losses['d_val'], 'discriminator validation', 'd_val_loss.png', log_dir)

    plot_loss(losses['p2p'], 'Pix2Pix', 'p2p_loss.png', log_dir)
    plot_loss(losses['p2p_val'], 'Pix2Pix validation', 'p2p_val_loss.png', log_dir)

    ###########################################################################
    #                          PLOT THE A->B RESULTS                          #
    ###########################################################################
    plt.figure(figsize=(10, 6))
    for i in range(N*N):
        a, _ = next(it_val)

        bp = atob.predict(a)
        img = compose_imgs(a[0], bp[0], is_a_binary=is_a_binary, is_b_binary=is_b_binary)

        plt.subplot(N, N, i+1)
        plt.imshow(img)
        plt.axis('off')

    plt.savefig(os.path.join(log_dir, 'atob.png'))
    plt.clf()

    # Make sure all the figures are closed.
    plt.close('all')


def save_weights(models, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Save the weights of the models into a file."""
    log_dir = get_log_dir(log_dir, expt_name)

    models.atob.save_weights(os.path.join(log_dir, ATOB_WEIGHTS_FILE), overwrite=True)
    models.d.save_weights(os.path.join(log_dir, D_WEIGHTS_FILE), overwrite=True)


def load_weights(atob, d, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Load the weights into the corresponding models."""
    log_dir = get_log_dir(log_dir, expt_name)

    atob.load_weights(os.path.join(log_dir, ATOB_WEIGHTS_FILE))
    d.load_weights(os.path.join(log_dir, D_WEIGHTS_FILE))


def load_weights_of(m, weights_file, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Load the weights of the model m."""
    log_dir = get_log_dir(log_dir, expt_name)

    m.load_weights(os.path.join(log_dir, weights_file))


def load_losses(log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Load the losses of the given experiment."""
    log_dir = get_log_dir(log_dir, expt_name)
    losses = pickle.load(open(os.path.join(log_dir, 'losses.pkl'), 'rb'))
    return losses


def load_params(params):
    """
    Load the parameters of an experiment and return them.

    The params passed as argument will be merged with the new params dict.
    If there is a conflict with a key, the params passed as argument prevails.
    """
    expt_dir = get_log_dir(params.log_dir, params.expt_name)

    expt_params = json.load(open(os.path.join(expt_dir, 'params.json'), 'rb'))

    # Update the loaded parameters with the current parameters. This will
    # override conflicting keys as expected.
    expt_params.update(params)

    return expt_params
