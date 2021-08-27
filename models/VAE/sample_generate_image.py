import os
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import warnings

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import *
from tensorflow.python.client import device_lib
import tensorflow as tf
warnings.filterwarnings('ignore')
K.clear_session()
device_lib.list_local_devices()



def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

img_shape = (128, 160, 1)
batch_size = 10
latent_dim = 2  # Dimensionality of the latent space: a plane

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
decoder = Model(decoder_input, x)
z_decoded = decoder(z)
y = CustomVariationalLayer()([input_img, z_decoded])
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

test=np.load('drive/My Drive/train_img.npy')
test = test[1:]
plt.imshow(test[0][100],'gray'),plt.show()

# make dataset
test_imgs =[]
for idx in range(test.shape[0]):
  for sec in range(test.shape[1]):
    test_img = test[idx][sec]
    test_imgs.append(test_img)
test_imgs = np.array(test_imgs).astype('float32')
print(test_imgs.shape)
print(test.min(), test.max())

test_dataset = test_imgs.reshape(len(test_imgs),128, 160,1)
print(test_dataset.shape)

tests=test_dataset[:7000]
test_val=test_dataset[7000:]
print(tests.shape, test_val.shape)

try:
  vae.fit(x=tests, y=None, epochs=10,
        batch_size=10, validation_data=(test_val, None))
finally:
      vae.save('drive/My Drive//vae_ep10.h5')

import matplotlib.pyplot as plt
from scipy.stats import norm
# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 128
figure = np.zeros((128 * n, 160 * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(128, 160)
        # plot
        plt.imshow(digit, cmap='Greys_r'),plt.show()
        figure[i * 128: (i + 1) * 128,
               j * 160: (j + 1) * 160] = digit


plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
