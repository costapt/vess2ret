__doc__ = """The model definitions for the pix2pix network taken from the
retina repository at https://github.com/costapt/vess2ret
"""
import os

import keras
from keras import backend as K
from keras import objectives
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

KERAS_2 = keras.__version__[0] == '2'
try:
    # keras 2 imports
    from keras.layers.convolutional import Conv2DTranspose
    from keras.layers.merge import Concatenate
except ImportError:
    print("Keras 2 layers could not be imported defaulting to keras1")
    KERAS_2 = False

K.set_image_dim_ordering('th')


def concatenate_layers(inputs, concat_axis, mode='concat'):
    if KERAS_2:
        assert mode == 'concat', "Only concatenation is supported in this wrapper"
        return Concatenate(axis=concat_axis)(inputs)
    else:
        return merge(inputs=inputs, concat_axis=concat_axis, mode=mode)


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    if KERAS_2:
        return Convolution2D(f,
                             kernel_size=(k, k),
                             padding=border_mode,
                             strides=(s, s),
                             **kwargs)
    else:
        return Convolution2D(f, k, k, border_mode=border_mode,
                             subsample=(s, s),
                             **kwargs)


def Deconvolution(f, output_shape, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    if KERAS_2:
        return Conv2DTranspose(f,
                               kernel_size=(k, k),
                               output_shape=output_shape,
                               strides=(s, s),
                               data_format=K.image_data_format(),
                               **kwargs)
    else:
        return Deconvolution2D(f, k, k, output_shape=output_shape,
                               subsample=(s, s), **kwargs)


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    if KERAS_2:
        return BatchNormalization(axis=axis, **kwargs)
    else:
        return BatchNormalization(mode=2,axis=axis, **kwargs)


def g_unet(in_ch, out_ch, nf, batch_size=1, is_binary=False, name='unet'):
    # type: (int, int, int, int, bool, str) -> keras.models.Model
    """Define a U-Net.

    Input has shape in_ch x 512 x 512
    Parameters:
    - in_ch: the number of input channels;
    - out_ch: the number of output channels;
    - nf: the number of filters of the first layer;
    - is_binary: if is_binary is true, the last layer is followed by a sigmoid
    activation function, otherwise, a tanh is used.
    >>> K.set_image_dim_ordering('th')
    >>> K.image_data_format()
    'channels_first'
    >>> unet = g_unet(1, 2, 3, batch_size=5, is_binary=True)
    TheanoShapedU-NET
    >>> for ilay in unet.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    >>> unet.summary()  #doctest: +NORMALIZE_WHITESPACE
     _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 1, 512, 512)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 3, 256, 256)       30
    _________________________________________________________________
    batch_normalization (BatchNo (None, 3, 256, 256)       12
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 3, 256, 256)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 6, 128, 128)       168
    _________________________________________________________________
    batch_normalization (BatchNo (None, 6, 128, 128)       24
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 6, 128, 128)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 12, 64, 64)        660
    _________________________________________________________________
    batch_normalization (BatchNo (None, 12, 64, 64)        48
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 12, 64, 64)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 32, 32)        2616
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 32, 32)        96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 32, 32)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 16, 16)        5208
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 16, 16)        96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 16, 16)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 8, 8)          5208
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 8, 8)          96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 8, 8)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 4, 4)          5208
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 4, 4)          96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 4, 4)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 2, 2)          5208
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 2, 2)          96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 2, 2)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 1, 1)          2328
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 1, 1)          96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 1, 1)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 2, 2)          2328
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 2, 2)          96
    _________________________________________________________________
    dropout (Dropout)            (None, 24, 2, 2)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 2, 2)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 2, 2)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 4, 4)          4632
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 4, 4)          96
    _________________________________________________________________
    dropout (Dropout)            (None, 24, 4, 4)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 4, 4)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 4, 4)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 8, 8)          4632
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 8, 8)          96
    _________________________________________________________________
    dropout (Dropout)            (None, 24, 8, 8)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 8, 8)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 8, 8)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 16, 16)        4632
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 16, 16)        96
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 16, 16)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 16, 16)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 32, 32)        4632
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 32, 32)        96
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 32, 32)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 32, 32)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 12, 64, 64)        2316
    _________________________________________________________________
    batch_normalization (BatchNo (None, 12, 64, 64)        48
    _________________________________________________________________
    concatenate (Concatenate)    (None, 24, 64, 64)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 64, 64)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 6, 128, 128)       582
    _________________________________________________________________
    batch_normalization (BatchNo (None, 6, 128, 128)       24
    _________________________________________________________________
    concatenate (Concatenate)    (None, 12, 128, 128)      0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 12, 128, 128)      0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 3, 256, 256)       147
    _________________________________________________________________
    batch_normalization (BatchNo (None, 3, 256, 256)       12
    _________________________________________________________________
    concatenate (Concatenate)    (None, 6, 256, 256)       0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 6, 256, 256)       0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 2, 512, 512)       50
    _________________________________________________________________
    activation (Activation)      (None, 2, 512, 512)       0
    =================================================================
    Total params: 51,809.0
    Trainable params: 51,197.0
    Non-trainable params: 612.0
    _________________________________________________________________
    >>> K.set_image_dim_ordering('tf')
    >>> K.image_data_format()
    'channels_last'
    >>> unet2=g_unet(3, 4, 2, batch_size=7, is_binary=False)
    TensorflowShapedU-NET
    >>> for ilay in unet2.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    >>> unet2.summary()  #doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 512, 512, 3)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 256, 256, 2)       56
    _________________________________________________________________
    batch_normalization (BatchNo (None, 256, 256, 2)       1024
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 256, 256, 2)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 128, 128, 4)       76
    _________________________________________________________________
    batch_normalization (BatchNo (None, 128, 128, 4)       512
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 128, 128, 4)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 64, 64, 8)         296
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64, 64, 8)         256
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 64, 64, 8)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 32, 32, 16)        1168
    _________________________________________________________________
    batch_normalization (BatchNo (None, 32, 32, 16)        128
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 32, 32, 16)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 16, 16)        2320
    _________________________________________________________________
    batch_normalization (BatchNo (None, 16, 16, 16)        64
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 16, 16, 16)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 8, 8, 16)          2320
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 16)          32
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 8, 8, 16)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 4, 4, 16)          2320
    _________________________________________________________________
    batch_normalization (BatchNo (None, 4, 4, 16)          16
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 4, 4, 16)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 2, 2, 16)          2320
    _________________________________________________________________
    batch_normalization (BatchNo (None, 2, 2, 16)          8
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 2, 2, 16)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 1, 1, 16)          1040
    _________________________________________________________________
    batch_normalization (BatchNo (None, 1, 1, 16)          4
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 1, 1, 16)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 2, 2, 16)          1040
    _________________________________________________________________
    batch_normalization (BatchNo (None, 2, 2, 16)          8
    _________________________________________________________________
    dropout (Dropout)            (None, 2, 2, 16)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 2, 2, 32)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 2, 2, 32)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 4, 4, 16)          2064
    _________________________________________________________________
    batch_normalization (BatchNo (None, 4, 4, 16)          16
    _________________________________________________________________
    dropout (Dropout)            (None, 4, 4, 16)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 4, 4, 32)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 4, 4, 32)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 8, 8, 16)          2064
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 16)          32
    _________________________________________________________________
    dropout (Dropout)            (None, 8, 8, 16)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 8, 8, 32)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 8, 8, 32)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 16, 16, 16)        2064
    _________________________________________________________________
    batch_normalization (BatchNo (None, 16, 16, 16)        64
    _________________________________________________________________
    concatenate (Concatenate)    (None, 16, 16, 32)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 16, 16, 32)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 32, 32, 16)        2064
    _________________________________________________________________
    batch_normalization (BatchNo (None, 32, 32, 16)        128
    _________________________________________________________________
    concatenate (Concatenate)    (None, 32, 32, 32)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 32, 32, 32)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 64, 64, 8)         1032
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64, 64, 8)         256
    _________________________________________________________________
    concatenate (Concatenate)    (None, 64, 64, 16)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 64, 64, 16)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 128, 128, 4)       260
    _________________________________________________________________
    batch_normalization (BatchNo (None, 128, 128, 4)       512
    _________________________________________________________________
    concatenate (Concatenate)    (None, 128, 128, 8)       0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 128, 128, 8)       0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 256, 256, 2)       66
    _________________________________________________________________
    batch_normalization (BatchNo (None, 256, 256, 2)       1024
    _________________________________________________________________
    concatenate (Concatenate)    (None, 256, 256, 4)       0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 256, 256, 4)       0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 512, 512, 4)       68
    _________________________________________________________________
    activation (Activation)      (None, 512, 512, 4)       0
    =================================================================
    Total params: 26,722.0
    Trainable params: 24,680.0
    Non-trainable params: 2,042.0
    _________________________________________________________________
    """
    merge_params = {
        'mode': 'concat',
        'concat_axis': 1
    }
    if K.image_dim_ordering() == 'th':
        print('TheanoShapedU-NET')
        i = Input(shape=(in_ch, 512, 512))

        def get_deconv_shape(samples, channels, x_dim, y_dim):
            return samples, channels, x_dim, y_dim

    elif K.image_dim_ordering() == 'tf':
        i = Input(shape=(512, 512, in_ch))
        print('TensorflowShapedU-NET')

        def get_deconv_shape(samples, channels, x_dim, y_dim):
            return samples, x_dim, y_dim, channels

        merge_params['concat_axis'] = 3
    else:
        raise ValueError(
            'Keras dimension ordering not supported: {}'.format(
                K.image_dim_ordering()))

    # in_ch x 512 x 512
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 256 x 256

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 128 x 128

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 32 x 32

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 16 x 16

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 8 x 8

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 4 x 4

    conv8 = Convolution(nf * 8)(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 2 x 2

    conv9 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv9 = BatchNorm()(conv9)
    x = LeakyReLU(0.2)(conv9)
    # nf*8 x 1 x 1

    dconv1 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 2, 2),
                           k=2, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    dconv1 = Dropout(0.5)(dconv1)

    x = concatenate_layers([dconv1, conv8], **merge_params)

    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 4, 4))(x)
    dconv2 = BatchNorm()(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = concatenate_layers([dconv2, conv7], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 8, 8))(x)
    dconv3 = BatchNorm()(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    x = concatenate_layers([dconv3, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 16, 16))(x)
    dconv4 = BatchNorm()(dconv4)
    x = concatenate_layers([dconv4, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 32, 32))(x)
    dconv5 = BatchNorm()(dconv5)
    x = concatenate_layers([dconv5, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 32 x 32

    dconv6 = Deconvolution(nf * 4,
                           get_deconv_shape(batch_size, nf * 4, 64, 64))(x)
    dconv6 = BatchNorm()(dconv6)
    x = concatenate_layers([dconv6, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(4 + 4) x 64 x 64

    dconv7 = Deconvolution(nf * 2,
                           get_deconv_shape(batch_size, nf * 2, 128, 128))(x)
    dconv7 = BatchNorm()(dconv7)
    x = concatenate_layers([dconv7, conv2], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(2 + 2) x 128 x 128

    dconv8 = Deconvolution(nf,
                           get_deconv_shape(batch_size, nf, 256, 256))(x)
    dconv8 = BatchNorm()(dconv8)
    x = concatenate_layers([dconv8, conv1], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(1 + 1) x 256 x 256

    dconv9 = Deconvolution(out_ch,
                           get_deconv_shape(batch_size, out_ch, 512, 512))(x)
    # out_ch x 512 x 512

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(dconv9)

    unet = Model(i, out, name=name)

    return unet


def discriminator(a_ch, b_ch, nf, opt=Adam(lr=2e-4, beta_1=0.5), name='d'):
    """Define the discriminator network.

    Parameters:
    - a_ch: the number of channels of the first image;
    - b_ch: the number of channels of the second image;
    - nf: the number of filters of the first layer.
    >>> K.set_image_dim_ordering('th')
    >>> disc=discriminator(3,4,2)
    >>> for ilay in disc.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    >>> disc.summary() #doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 7, 512, 512)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 2, 256, 256)       128
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 2, 256, 256)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 4, 128, 128)       76
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 4, 128, 128)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 8, 64, 64)         296
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 8, 64, 64)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 32, 32)        1168
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 16, 32, 32)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 1, 16, 16)         145
    _________________________________________________________________
    activation (Activation)      (None, 1, 16, 16)         0
    =================================================================
    Total params: 1,813.0
    Trainable params: 1,813.0
    Non-trainable params: 0.0
    _________________________________________________________________
    """
    i = Input(shape=(a_ch + b_ch, 512, 512))

    # (a_ch + b_ch) x 512 x 512
    conv1 = Convolution(nf)(i)
    x = LeakyReLU(0.2)(conv1)
    # nf x 256 x 256

    conv2 = Convolution(nf * 2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 128 x 128

    conv3 = Convolution(nf * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 8)(x)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 32 x 32

    conv5 = Convolution(1)(x)
    out = Activation('sigmoid')(conv5)
    # 1 x 16 x 16

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=opt, loss=d_loss)
    return d


def pix2pix(atob, d, a_ch, b_ch, alpha=100, is_a_binary=False,
            is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5), name='pix2pix'):
    # type: (...) -> keras.models.Model
    """
    Define the pix2pix network.
    :param atob:
    :param d:
    :param a_ch:
    :param b_ch:
    :param alpha:
    :param is_a_binary:
    :param is_b_binary:
    :param opt:
    :param name:
    :return:
    >>> K.set_image_dim_ordering('th')
    >>> unet = g_unet(3, 4, 2, batch_size=8, is_binary=False)
    TheanoShapedU-NET
    >>> disc=discriminator(3,4,2)
    >>> pp_net=pix2pix(unet, disc, 3, 4)
    >>> for ilay in pp_net.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    >>> pp_net.summary()  #doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 3, 512, 512)       0
    _________________________________________________________________
     (Model)                     (None, 4, 512, 512)       23454
    _________________________________________________________________
    concatenate (Concatenate)    (None, 7, 512, 512)       0
    _________________________________________________________________
     (Model)                     (None, 1, 16, 16)         1813
    =================================================================
    Total params: 25,267.0
    Trainable params: 24,859.0
    Non-trainable params: 408.0
    _________________________________________________________________
    """
    a = Input(shape=(a_ch, 512, 512))
    b = Input(shape=(b_ch, 512, 512))

    # A -> B'
    bp = atob(a)

    # Discriminator receives the pair of images
    d_in = concatenate_layers([a, bp], mode='concat', concat_axis=1)

    pix2pix = Model([a, b], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = objectives.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        return L_adv + alpha * L_atob

    # This network is used to train the generator. Freeze the discriminator part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)
    return pix2pix


if __name__ == '__main__':
    import doctest

    TEST_TF = True
    if TEST_TF:
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    else:
        os.environ['KERAS_BACKEND'] = 'theano'
    doctest.testsource('models.py', verbose=True, optionflags=doctest.ELLIPSIS)
