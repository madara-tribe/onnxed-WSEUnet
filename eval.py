from tensorflow.keras.optimizers import RMSprop, Adam, SGD

import numpy as np
import os
import math
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *


from loss.bce_dice_loss import bce_dice_loss
from Augmentation import sift_angle, flip
from models.Unet.WSEUnet import create_model


EPOCHS = 100
H=448
W=448
BATCH_SIZE=4

class trainGenerator(object):
    def __init__(self):
        self.img_generator=[]
        self.anno_generator=[]

    def reset(self):
        self.img_generator=[]
        self.anno_generator=[]

    def flow(self, imgs, annos, batch=4):
        while True:
            for anno, img in zip(annos, imgs):
                self.img_generator.append(img)
                self.anno_generator.append(anno)
                if len(self.anno_generator)==4:
                    input_img = np.array(self.img_generator)
                    input_anno = np.array(self.anno_generator)
                    self.reset()
                    yield input_img.reshape(4, H, W, 4), input_anno.reshape(4, H, W, 1)

def load_data(split_n=50):
    imgs, annos = np.load('data/test_sar_img.npy'), np.load('data/test_sar_anno.npy')
    imgs, annos = imgs.astype(np.float32), annos.astype(np.float32)
    print(imgs.shape, annos.shape)
    return imgs, annos

def load_model(weight_path=None):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    models = create_model(1, input_shape=(H, W, 4))
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.compile(optimizer=adam, loss=bce_dice_loss, metrics=['accuracy'])
    models.summary()
    return models


def train():
    X_test, y_test = load_data(split_n=50)
    models = load_model(weight_path='cp-0003.ckpt')
    pred = models.predict(X_test) 
    #_, acc = models.evaluate(X_test, y_test, verbose=0)
    print(pred.shape)
    

if __name__=='__main__':
    train()


