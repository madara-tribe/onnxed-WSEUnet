from tensorflow.keras.optimizers import RMSprop, Adam, SGD

import numpy as np
import os, cv2
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from losses.bce_dice_loss import bce_dice_loss, dice_loss
from losses.focal_loss import focal_loss
from fbeta_score import binary_fbeta
from Augmentation import sift_angle, flip, random_eraser
from models.WSEUnet import create_model
from models.res50_unet import ResNet_Unet 
from option_parser import get_option
H=448
W=448

def gene_flow_inputs(datagen, x_image, y_train, batch=2):
    batch = datagen.flow(x_image, y_train, batch_size=batch)
    while True:
        batch_image, batch_mask = batch.next()
        yield batch_image, batch_mask


def load_data(config=None, split_n=2):
    generators = ImageDataGenerator(rotation_range=10,
                                    horizontal_flip=True,
                                    #vertical_flip=True,
                                    fill_mode='constant', 
                                    validation_split=0.2)
                                    #preprocessing_function=random_eraser.get_random_eraser(v_l=0, v_h=1))
    imgs, annos = np.load('data/sar_img.npy'), np.load('data/sar_anno.npy')
    
    imgs, annos = imgs.astype(np.float32), annos.astype(np.float32)
    #imgs1, annos1 = imgs[:, ::-1], annos[:, ::-1]
    #imgs= np.vstack([imgs, imgs1])
    #annos = np.vstack([annos, annos1])
    X_train, y_train = imgs[split_n:], annos[split_n:]
    X_val, y_val = imgs[:split_n], annos[:split_n]
    print(imgs.shape, annos.shape)
    training = gene_flow_inputs(generators, X_train, y_train, batch=config.batch_size)
    return training, X_train, X_val, y_val

def load_model(config=None, weight_path=None):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    if config.model_type=='se':
        models = create_model(config.num_cls, input_shape=(H, W, 3))
        models.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    elif config.model_type=='resnet':
        models = ResNet_Unet(nClasses=config.num_cls, input_shape=(H, W, 3))
        models.compile(optimizer=adam, loss=bce_dice_loss, metrics=[binary_fbeta])
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.summary()
    return models


def train(cfg):
    training, X_train, X_val, y_val = load_data(cfg, split_n=5)
    models = load_model(config=cfg, weight_path=None)
     
    checkpoint_path = "weights/cp-{epoch:02d}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1)
    tb = TensorBoard(log_dir='./logs')
    callback = [cp_callback, reduce_lr, tb]
    
    print('train')
    startTime1 = datetime.now() #DB
    hist1 = models.fit(training, steps_per_epoch=int(len(X_train)/cfg.batch_size)*10, epochs=cfg.epoch, batch_size=cfg.batch_size, validation_data=(X_val, y_val), callbacks=callback, verbose=1)

    endTime1 = datetime.now()
    diff1 = endTime1 - startTime1
    print("\n")
    print("Elapsed time for Keras training (s): ", diff1.total_seconds())
    print("\n")

    _, acc = models.evaluate(X_val, y_val, verbose=0)
    print('\nTest accuracy: {0}'.format(acc))
    

if __name__=='__main__':
    cfg = get_option()
    os.makedirs(cfg.weight_dir, exist_ok=True)
    train(cfg)
