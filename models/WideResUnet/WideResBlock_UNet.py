from models.WideResUnet.ResBlock import dw_conv, res_block, up_conv
import keras
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras import layers

def create_model(nClass, input_shape=(256,256,3), regularizers = None, k=1, lr=1e-3):
    inputs = Input(shape=input_shape)
    if regularizers:
        KL_REG = keras.regularizers.l2(0.0001)
    else:
        KL_REG = None

    i = 0

    nb_filter = [16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16]

    #0
    x = Conv2D(nb_filter[i] *k, (3, 3), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter[i] *k, (3, 3), padding='same', use_bias=False)(x)
    x0 = BatchNormalization()(x)
    x = Activation('relu')(x0)
    i += 1

    #1
    x = dw_conv(x0, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x1 = res_block(x, k, nb_filter[i])
    i += 1

    #2
    x = dw_conv(x1, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x2 = res_block(x, k, nb_filter[i])
    i += 1

    #3
    x = dw_conv(x2, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x3 = res_block(x, k, nb_filter[i])
    i += 1

    #4
    x = dw_conv(x3, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x4 = res_block(x, k, nb_filter[i])
    i += 1

    #--------------- center ------------
    x = dw_conv(x4, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x = res_block(x, k, nb_filter[i])
    #--------------- center ------------
    i += 1

    #4
    x = up_conv(x, x4, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x = res_block(x, k, nb_filter[i])
    i += 1

    #3
    x = up_conv(x, x3, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x = res_block(x, k, nb_filter[i])
    i += 1

    #2
    x = up_conv(x, x2, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x = res_block(x, k, nb_filter[i])
    i += 1

    #1
    x = up_conv(x, x1, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x = res_block(x, k, nb_filter[i])
    i += 1

    #0
    x = up_conv(x, x0, nb_filter[i], k, kl_reg = KL_REG)
    x = res_block(x, k, nb_filter[i])
    x = res_block(x, k, nb_filter[i])
    x = Activation('relu')(x)

    classify = Conv2D(nClass, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=classify)
    return model
