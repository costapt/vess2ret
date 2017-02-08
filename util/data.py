"""Auxiliar methods to deal with loading the dataset."""
import os
import random

import numpy as np

from keras.preprocessing.image import apply_transform, flip_axis
from keras.preprocessing.image import transform_matrix_offset_center
from keras.preprocessing.image import Iterator, load_img, img_to_array


class TwoImageIterator(Iterator):
    """Class to iterate A and B images at the same time."""

    def __init__(self, directory, a_dir_name='A', b_dir_name='B', load_to_memory=False,
                 is_a_binary=False, is_b_binary=False, is_a_grayscale=False,
                 is_b_grayscale=False, target_size=(256, 256), rotation_range=0.,
                 height_shift_range=0., width_shift_range=0., zoom_range=0.,
                 fill_mode='constant', cval=0., horizontal_flip=False,
                 vertical_flip=False,  dim_ordering='default', N=-1,
                 batch_size=32, shuffle=True, seed=None):
        """
        Iterate through two directories at the same time.

        Files under the directory A and B with the same name will be returned
        at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - load_to_memory: if true, loads the images to memory when creating the
        iterator;
        - is_a_binary: converts A images to binary images. Applies a threshold of 0.5.
        - is_b_binary: converts B images to binary images. Applies a threshold of 0.5.
        - is_a_grayscale: if True, A images will only have one channel.
        - is_b_grayscale: if True, B images will only have one channel.
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator.
        """
        self.directory = directory

        self.a_dir = os.path.join(directory, a_dir_name)
        self.b_dir = os.path.join(directory, b_dir_name)

        a_files = set(x for x in os.listdir(self.a_dir))
        b_files = set(x for x in os.listdir(self.b_dir))
        # Files inside a and b should have the same name. Images without a pair are discarded.
        self.filenames = list(a_files.intersection(b_files))

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:N]
        self.N = len(self.filenames)
        if self.N == 0:
            raise Exception("""Did not find any pair in the dataset. Please check that """
                            """the names and extensions of the pairs are exactly the same. """
                            """Searched inside folders: {0} and {1}""".format(self.a_dir, self.b_dir))

        self.dim_ordering = dim_ordering
        if self.dim_ordering not in ('th', 'default', 'tf'):
            raise Exception('dim_ordering should be one of "th", "tf" or "default". '
                            'Got {0}'.format(self.dim_ordering))

        self.target_size = target_size

        self.is_a_binary = is_a_binary
        self.is_b_binary = is_b_binary
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale

        self.image_shape_a = self._get_image_shape(self.is_a_grayscale)
        self.image_shape_b = self._get_image_shape(self.is_b_grayscale)

        self.load_to_memory = load_to_memory
        if self.load_to_memory:
            self._load_imgs_to_memory()

        if self.dim_ordering in ('th', 'default'):
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

        super(TwoImageIterator, self).__init__(len(self.filenames), batch_size,
                                               shuffle, seed)

    def _get_image_shape(self, is_grayscale):
        """Auxiliar method to get the image shape given the color mode."""
        if is_grayscale:
            if self.dim_ordering == 'tf':
                return self.target_size + (1,)
            else:
                return (1,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                return self.target_size + (3,)
            else:
                return (3,) + self.target_size

    def _load_imgs_to_memory(self):
        """Load images to memory."""
        if not self.load_to_memory:
            raise Exception('Can not load images to memory. Reason: load_to_memory = False')

        self.a = np.zeros((self.N,) + self.image_shape_a)
        self.b = np.zeros((self.N,) + self.image_shape_b)

        for idx in range(self.N):
            ai, bi = self._load_img_pair(idx, False)
            self.a[idx] = ai
            self.b[idx] = bi

    def _binarize(self, batch):
        """Make input binary images have 0 and 1 values only."""
        bin_batch = batch / 255.
        bin_batch[bin_batch >= 0.5] = 1
        bin_batch[bin_batch < 0.5] = 0
        return bin_batch

    def _normalize_for_tanh(self, batch):
        """Make input image values lie between -1 and 1."""
        tanh_batch = batch - 127.5
        tanh_batch /= 127.5
        return tanh_batch

    def _load_img_pair(self, idx, load_from_memory):
        """Get a pair of images with index idx."""
        if load_from_memory:
            a = self.a[idx]
            b = self.b[idx]
            return a, b

        fname = self.filenames[idx]

        a = load_img(os.path.join(self.a_dir, fname),
                     grayscale=self.is_a_grayscale,
                     target_size=self.target_size)
        b = load_img(os.path.join(self.b_dir, fname),
                     grayscale=self.is_b_grayscale,
                     target_size=self.target_size)

        a = img_to_array(a, self.dim_ordering)
        b = img_to_array(b, self.dim_ordering)

        return a, b

    def _random_transform(self, a, b):
        """
        Random dataset augmentation.

        Adapted from https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        """
        # a and b are single images, so they don't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * a.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * a.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(rotation_matrix, translation_matrix), zoom_matrix)

        h, w = a.shape[img_row_index], a.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        a = apply_transform(a, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        b = apply_transform(b, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_col_index)
                b = flip_axis(b, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_row_index)
                b = flip_axis(b, img_row_index)

        return a, b

    def next(self):
        """Get the next pair of the sequence."""
        # Lock the iterator when the index is changed.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)

        batch_a = np.zeros((current_batch_size,) + self.image_shape_a)
        batch_b = np.zeros((current_batch_size,) + self.image_shape_b)

        for i, j in enumerate(index_array):
            a_img, b_img = self._load_img_pair(j, self.load_to_memory)
            a_img, b_img = self._random_transform(a_img, b_img)

            batch_a[i] = a_img
            batch_b[i] = b_img

        if self.is_a_binary:
            batch_a = self._binarize(batch_a)
        else:
            batch_a = self._normalize_for_tanh(batch_a)

        if self.is_b_binary:
            batch_b = self._binarize(batch_b)
        else:
            batch_b = self._normalize_for_tanh(batch_b)

        return [batch_a, batch_b]
