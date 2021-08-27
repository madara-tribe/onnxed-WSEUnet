from PIL import Image, ImageDraw
import numpy as np
import os
import math
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from VAE import VAE
from HistoryCallbackLoss import HistoryCheckpoint
from tensorflow.python.client import device_lib
device_lib.list_local_devices()



# load image
class trainGenerator(object):
    def __init__(self):
        self.img_generator=[]

    def flow_from_dir(self, img_path):
        for imgs in os.listdir(img_path):
            imgs=cv2.imread(img_path+'/'+imgs)

            if imgs is not None:
                #imgs = cv2.resize(imgs, (1024, 1024))
                self.img_generator.append(imgs/ 255)
        input_img=np.array(self.img_generator, dtype=np.float32)
        return input_img


# load image (already crop and resize)
def train():
    train=trainGenerator()
    x_train = train.flow_from_dir('VAE/vae_train')
    print(x_train.shape)
    #plt.imshow(x_train [0]), plt.show()
    
    valid = trainGenerator()
    x_valid = valid.flow_from_dir('VAE/vae_valid')
    print(x_valid.shape)
    #plt.imshow(x_valid[0]), plt.show()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    def step_decay(epoch):
        initial_lrate = 0.0001
        decay_rate = 0.5
        decay_steps = 8.0
        lrate = initial_lrate * math.pow(decay_rate,
               math.floor((1+epoch)/decay_steps))
        return lrate


    callback=[]
    callback.append(HistoryCheckpoint(filepath='tb/LearningCurve_{history}.png', verbose=1, period=300))
    callback.append(LearningRateScheduler(step_decay))

    print('call model')
    model = VAE()
    model, loss=model.vae_net()
    #model.load_weights("logss/10_unet.hdf5")

    model.add_loss(loss)
    model.compile(optimizer =Adam(lr=0.0001))
    model.summary()
    
    print('train')
    try:
        model.fit(train, batch_size=20, epochs=300,
              callbacks=callback, validation_data=(valid, None))
    finally:
        model.save('/Users/desktop/vae_model.h5')
