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
from tensorflow.keras import metrics

from models.efficientnet.efficientunet import _get_efficient_unet
from models.efficientnet.efficientnet import get_efficientnet_b5_encoder
from augmentation import sift_angle, flip, random_eraser
from models.SEUNet.WSEUnet import create_model
from models.mish_Resnet_RS.Resnet_RS import create_model as create_mishrs_model
from option_parser import get_option
from fbeta_score import binary_fbeta 
H=448
W=448
C=3

def gene_flow_inputs(datagen, x_image, y_train, batch=2):
    batch = datagen.flow(x_image, y_train, batch_size=batch)
    while True:
        batch_image, batch_mask = batch.next()
        yield batch_image, batch_mask

def flipaug(X, y):
    X1, y1 = flip.npflip(X, types='lr'), flip.npflip(y, types='lr')
    X2, y2 = flip.npflip(X, types='up'), flip.npflip(y, types='up')
    X3, y3 = flip.npflip(X, types='lrup'), flip.npflip(y, types='lrup')
    X, y = np.vstack([X, X1, X2, X3]), np.vstack([y, y1, y2, y3])
    return X, y

def load_data(config=None):
    imgs, annos = np.load('data/sar_img.npy'), np.load('data/sar_anno.npy')
    N = 5
    imgs, annos = imgs.astype(np.float32), annos.astype(np.float32)
    X_train, y_train = imgs[N:], annos[N:]
    X_val, y_val = imgs[:N], annos[:N]
    X_train, y_train = flipaug(X_train, y_train)
    X_val, y_val = flipaug(X_val, y_val)
    print(imgs.shape, annos.shape)
    return X_train, y_train, X_val, y_val

def load_model(config=None, weight_path=None):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    if config.model_type=='se':
        models = create_model(config.num_cls, input_shape=(H, W, C))
        models.compile(optimizer=adam, loss='mean_absolute_error', metrics=['accuracy', metrics.mae, binary_fbeta])
    elif config.model_type=='rs_mish':
        models = create_mishrs_model(config.num_cls, input_shape=(H, W, C))
        models.compile(optimizer=adam, loss='mean_absolute_error', metrics=['accuracy', metrics.mae, binary_fbeta])
    elif config.model_type=='efficientunet':
        encoder = get_efficientnet_b5_encoder(input_shape=(H, W, C), pretrained=True)
        models = _get_efficient_unet(encoder, out_channels=config.num_cls,
                          concat_input=True, fpa=None, hypercolumn=None)
        models.compile(optimizer=adam, loss='mean_absolute_error', metrics=['accuracy', metrics.mae, binary_fbeta])
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.summary()
    return models


def train(cfg):
    X_train, y_train, X_val, y_val = load_data(cfg)
    models = load_model(config=cfg, weight_path=None)

    checkpoint_path = "weights/cp-{epoch:02d}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1)
    tb = TensorBoard(log_dir='./logs')
    callback = [cp_callback, reduce_lr, tb]
    
    print('train')
    startTime1 = datetime.now() #DB
    hist1 = models.fit(X_train, y_train, epochs=cfg.epoch, batch_size=cfg.batch_size, validation_data=(X_val, y_val), callbacks=callback, verbose=1)

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
