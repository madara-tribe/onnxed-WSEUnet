from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import *
import keras
from keras import backend as K
from keras.models import Model
import tensorflow as tf
from keras import optimizers
IMAGE_ORDERING =  "channels_last"
def up_conv(last_layer, channel_num):
    up7 = UpSampling2D(size=(2, 2),data_format=IMAGE_ORDERING, interpolation="bilinear")(last_layer) #56x56x256
    u7 = Conv2D(channel_num, kernel_size=2, data_format=IMAGE_ORDERING,
                activation="relu", padding="same", kernel_initializer="he_normal")(up7)
    return u7

def res_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, activation=True):
    #firt layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
               kernel_initializer = 'he_normal', padding = 'same', data_format=IMAGE_ORDERING)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
               kernel_initializer = 'he_normal', padding = 'same', data_format=IMAGE_ORDERING)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)

    return x

def upsampling_step(skipped_conv, num_output_filters, prev_conv = None):
    num_filters = skipped_conv.output_shape[-1]
    if prev_conv != None:
        concat_layer = concatenate([skipped_conv.output, prev_conv])
    else:
        concat_layer = skipped_conv.output
    print('concat_layer', concat_layer)
    up = up_conv(concat_layer, num_filters)
    print('up', up)
    up = res_block(up, num_filters)
    return up


def encoder_R50(input_tensor, trained_weight=True):
    encoder_R50 = ResNet50(include_top = False, weights = 'imagenet', input_tensor = input_tensor, classes=1)
    encoder_R50.layers.pop()
    #for layer in encoder_R50.layers:
      #layer.trainable = False

    # Layers from ResNet50 to make skip connections
    skip_ix = [172, 140, 78, 36, 3]
    # Layers in decoder to connect to encoder
    skip_end = []
    for i in skip_ix:
        skip_end.append(encoder_R50.layers[i])
    return skip_end



def ResNet_Unet(nClasses, input_shape=(256, 256, 3), trained_weight=True):
    input_image = Input(shape=input_shape)
    encoder1_out = encoder_R50(input_tensor=input_image, trained_weight=True)
    
    
    conv_layer = upsampling_step(encoder1_out[0], 1024)
    conv_layer = upsampling_step(encoder1_out[1], 512, conv_layer)
    conv_layer = upsampling_step(encoder1_out[2], 256, conv_layer)
    conv_layer = upsampling_step(encoder1_out[3], 64, conv_layer)
    conv_layer = upsampling_step(encoder1_out[4], 64, conv_layer)
    o = Conv2D(nClasses, 1, padding = 'same', activation = 'sigmoid', kernel_initializer = 'he_normal', name = 'edges')(conv_layer)
    #edges = output(conv2, False)
    model = Model(inputs = input_image, outputs = o)
    return model
