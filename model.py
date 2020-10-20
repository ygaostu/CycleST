import numpy as np
import tensorflow as tf
import scipy.io as sio
from tensorflow.keras import layers
import sys, math


def load_shearlet_system(path, height, width):
    try:
        fmat = sio.loadmat(path)
    except FileNotFoundError:
        print(f"\n Could not find shearlet system file: {path} \n")
        sys.exit()
        
    dec = fmat["dec"].astype(np.float32)
    rec = fmat["rec"].astype(np.float32)
    ksize, _, nfilter = dec.shape 
    assert ksize <= height
    assert ksize <= width

    row_begin = int(math.ceil((height-ksize)/2.))
    col_begin = int(math.ceil((width-ksize)/2.))
    
    # Numpy -> Tensorflow 
    dec = tf.transpose(dec, (2, 0, 1))
    rec = tf.transpose(rec, (2, 0, 1))
    
    dec = tf.signal.fftshift(dec, (1, 2))
    rec = tf.signal.fftshift(rec, (1, 2))
    
    paddings = tf.constant([[0, 0], [row_begin, height-ksize-row_begin], [col_begin, width-ksize-col_begin]])
    dec_padd = tf.pad(dec, paddings, "CONSTANT")
    rec_padd = tf.pad(rec, paddings, "CONSTANT")

    dec_padd = tf.signal.ifftshift(dec_padd, (1, 2))
    rec_padd = tf.signal.ifftshift(rec_padd, (1, 2))
    
    return tf.signal.fft2d(tf.cast(dec_padd, tf.complex64)), tf.signal.fft2d(tf.cast(rec_padd, tf.complex64))


class AnalysisTrans(layers.Layer):
    """[batch*ch, 1, h, w] -> [batch*ch, 68, h, w]"""

    def __init__(self,
                dec_fft,
                name='analysis_transform',
                **kwargs):
        super(AnalysisTrans, self).__init__(name=name, **kwargs)
        self.dec_fft = dec_fft

    def call(self, inputs):
        x = tf.cast(inputs, tf.complex64)
        coeffs = tf.signal.ifft2d(tf.multiply(tf.signal.fft2d(x), self.dec_fft) ) 
        return tf.cast(coeffs, tf.float32)


class SynthesisTrans(layers.Layer):
    """[batch*ch, 68, h, w] -> [batch*ch, h, w]"""

    def __init__(self,
                rec_fft,
                name='synthesis_transform',
                **kwargs):
        super(SynthesisTrans, self).__init__(name=name, **kwargs)
        self.rec_fft = rec_fft

    def call(self, inputs):
        coeffs = tf.cast(inputs, tf.complex64)
        coeffs_fft = tf.multiply(tf.signal.fft2d(coeffs), self.rec_fft ) 
        x = tf.signal.ifft2d(tf.reduce_sum(coeffs_fft, -3))
        return tf.cast(x, tf.float32)


def conv_layer(filters, size):

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
    )
    result.add(tf.keras.layers.LeakyReLU())
    return result


def downsample(filters, size):

    result = conv_layer(filters, size)
    result.add(
        tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_first')
    )
    return result


def upsample(filters, size):

    result = conv_layer(filters, size)
    result.add(
        tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')
    )
    return result


def Model(path_model, IMG_HEIGHT, IMG_WIDTH, ST_CHANNELS=68):

    dec_fft, rec_fft = load_shearlet_system(path_model, IMG_HEIGHT, IMG_WIDTH)
    ana = AnalysisTrans(dec_fft)
    syn = SynthesisTrans(rec_fft)

    down_stack = [
        downsample(96, 3), 
        downsample(128, 3), 
        downsample(128, 3), 
        downsample(128, 3),
    ]

    up_stack = [
        upsample(128, 3),
        upsample(128, 3),
        upsample(96, 3), 
    ]

    concat = tf.keras.layers.Concatenate(axis=-3)


    ssepi = tf.keras.layers.Input(shape=[3, IMG_HEIGHT, IMG_WIDTH]) # b, c, h, w
    x = tf.reshape(ssepi, [-1, 1, IMG_HEIGHT, IMG_WIDTH]) # b*3, 1, h, w 

    input = ana(x)
    x = input

    x = conv_layer(ST_CHANNELS, 3)(x)

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = upsample(96, 3)(x) 
    
    
    x = tf.keras.layers.Conv2D(filters=ST_CHANNELS, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters=ST_CHANNELS, kernel_size=1, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(x)
    x = (x+input)


    x = tf.reshape(x, [-1, 3, ST_CHANNELS, IMG_HEIGHT, IMG_WIDTH])  # b, c, 68, h, w
    rec_dsepi = syn(x)

    return tf.keras.Model(inputs=ssepi, outputs=rec_dsepi)
