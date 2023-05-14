import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow import nn
from .functional_layers import *

def DCGANGenerator(
        noise_size : int,
        output_size : tuple,
        output_channels : int,
        name=None
):
    """DCGAN Generator
    Upscales 3 times from noise to output_size
    
    Parameters
    ----------
    noise_size : int
        size of the noise vector
    output_size : tuple
        output size (H,W)
        Assert that output_size[0] and output_size[1] are divisible by 8
    output_channels : int
        number of output channels
    """
    noise = keras.Input(shape=(noise_size,), name=nc(name,'noise'))
    first_layer_size = (output_size[0]//8, output_size[1]//8)
    assert first_layer_size[0]*8 == output_size[0], "output_size[0] must be divisible by 8"
    assert first_layer_size[1]*8 == output_size[1], "output_size[1] must be divisible by 8"
    x = layers.Dense(256*first_layer_size[0]*first_layer_size[1])(noise)
    x = layers.Reshape((first_layer_size[0],first_layer_size[1],256))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(128, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(output_channels, 5, strides=1, padding='same')(x)
    x = layers.Activation('tanh')(x) 

    return keras.Model(noise, x, name=nc(name,'generator'))

def DCGANDiscriminator(
        input_size : tuple,
        name=None
):
    """DCGAN Discriminator
    """
    inputs = keras.Input(shape=input_size, name=nc(name,'input'))
    x = layers.Conv2D(32, 5, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return keras.Model(inputs, x, name=nc(name,'discriminator'))

    