from tensorflow.keras.optimizers import RMSprop, Adam, SGD

import numpy as np
import os, cv2
import math
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *


from loss.bce_dice_loss import bce_dice_loss
from Augmentation import sift_angle, flip
from models.WSEUNet.WSEUnet import create_model
from option_parser import get_option

H=448
W=448

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
                if len(self.anno_generator)==batch:
                    input_img = np.array(self.img_generator)
                    input_anno = np.array(self.anno_generator)
                    self.reset()
                    yield input_img.reshape(batch, H, W, 4), input_anno.reshape(batch, H, W, 1)

def load_data(split_n=50):
    imgs, annos = np.load('data/sar_img.npy'), np.load('data/sar_anno.npy')
    imgs, annos = imgs.astype(np.float32), annos.astype(np.float32)
    X_train, y_train = imgs[split_n:], annos[split_n:]
    X_val, y_val = imgs[:split_n], annos[:split_n]
    print(imgs.shape, annos.shape)
    return X_train, y_train, X_val, y_val

def load_model(num_cls, weight_path=None):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    models = create_model(num_cls, input_shape=(H, W, 4))
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.compile(optimizer=adam, loss=bce_dice_loss, metrics=['accuracy'])
    models.summary()
    return models


def train(cfg):
    X_train, y_train, X_val, y_val = load_data(split_n=50)
    gene = trainGenerator().flow(X_train, y_train, batch=cfg.batch_size)

    models = load_model(num_cls=cfg.num_cls, weight_path=None)
     
    checkpoint_path = "weights/cp-{epoch:02d}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1)
    tb = TensorBoard(log_dir='./logs')
    callback = [cp_callback, reduce_lr, tb]
    
    print('train')
    startTime1 = datetime.now() #DB
    hist1 = models.fit(gene, steps_per_epoch=int(len(X_train)/cfg.batch_size), epochs=cfg.epoch, validation_data=(X_val, y_val), batch_size=cfg.batch_size, callbacks=callback, verbose=1)

    endTime1 = datetime.now()
    diff1 = endTime1 - startTime1
    print("\n")
    print("Elapsed time for Keras training (s): ", diff1.total_seconds())
    print("\n")

    _, acc = models.evaluate(X_val, y_val, verbose=0)
    print('\nTest accuracy: {0}'.format(acc))
    

if __name__=='__main__':
    cfg = get_option()
    train(cfg)


