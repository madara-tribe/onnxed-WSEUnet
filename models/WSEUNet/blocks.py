import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np

def dw_conv(init, nb_filter, k, kl_reg = None):
    residual = Conv2D(nb_filter * k, (1, 1), strides=(2, 2), padding='same', use_bias=False)(init)
    residual = x = BatchNormalization()(residual)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(init)
    x = BatchNormalization()(x)
    x = tfa.activations.mish(x)
    x = Dropout(0.4)(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])

    return x

def up_conv(init, skip, nb_filter, k, kl_reg = None):
    x = Conv2DTranspose(nb_filter * k, (3, 3), padding='same', strides=(2, 2), kernel_regularizer=kl_reg)(init)
    x = BatchNormalization()(x)
    x = layers.add([x, skip])
    return x

def res_block(init, nb_filter, k=1):
    x = tfa.activations.mish(init)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tfa.activations.mish(x)
    x = Dropout(0.4)(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Squeeze_excitation_layer(x)

    x = layers.add([init, x])
    return x


def Squeeze_excitation_layer(input_x):
    ratio = 4
    out_dim =  int(np.shape(input_x)[-1])
    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = Dense(units=int(out_dim / ratio))(squeeze)
    excitation = tfa.activations.mish(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = layers.Reshape([-1,1,out_dim])(excitation)
    scale = layers.multiply([input_x, excitation])

    return scale
