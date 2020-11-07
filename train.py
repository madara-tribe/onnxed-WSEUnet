import keras
from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import binary_crossentropy
from keras import backend as K
from keras import layers
import numpy as np
import os
import math
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator

from augumention import sift_angle, flips
from models.WideResUnet.WideResBlock_UNet import create_model


XPATH = 'train_data'
YPATH = 'valid_data'
EPOCHS = 1100
H=256
W=256

class trainGenerator(object):
    def __init__(self):
        self.img_generator=[]

    def flow_from_dir(self, img_path):
        for imgs in os.listdir(img_path):
            if '.DS_Store' in imgs:
                pass
            imgs=cv2.imread(img_path+'/'+imgs)

            if imgs is not None:
                imgs = cv2.resize(imgs, (W, H), cv2.INTER_NEAREST)
                self.img_generator.append(imgs/255)
        input_img=np.array(self.img_generator, dtype=np.float32)
        return input_img
  
def train_fit(model, X, Y, valX, valY):
    print('train')
    startTime1 = datetime.now() #DB
    hist1 = model.fit(X, Y, epochs=EPOCHS, validation_data=(valX, valY))
    endTime1 = datetime.now()
    diff1 = endTime1 - startTime1
    print("\n")
    print("Elapsed time for Keras training (s): ", diff1.total_seconds())
    print("\n")

    for key in ["loss", "val_loss"]:
        plt.plot(hist1.history[key],label=key)
    plt.legend()

def train_generator(model, X, Y, valX, valY):
    def gene_flow_inputs(datagen, x_image, y_train, batch=2):
        batch = datagen.flow(x_image, y_train, batch_size=batch)
        while True:
            batch_image, batch_mask = batch.next()
            yield batch_image, batch_mask

    generators = ImageDataGenerator(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    training = gene_flow_inputs(generators, X, Y)

    print('train')
    startTime1 = datetime.now() #DB
    hist1 = model.fit_generator(training, steps_per_epoch=3000, epochs=10, epochs=EPOCHS, validation_data=(valX, valY))
    endTime1 = datetime.now()
    diff1 = endTime1 - startTime1
    print("\n")
    print("Elapsed time for Keras training (s): ", diff1.total_seconds())
    print("\n")

    for key in ["loss", "val_loss"]:
        plt.plot(hist1.history[key],label=key)
    plt.legend()


def predict(model, X):
    pred = model.predict(X)
    print(pred.shape)
    plt.imshow(X[0]),plt.show()
    plt.imshow(pred[0]),plt.show()


def main():
    model = create_model(nClass=3, input_shape=(H,W,3))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics='accuracy')
    model.summary()


    train=trainGenerator()
    X_train = train.flow_from_dir(XPATH)
    print(X_train.shape)
    #plt.imshow(X_train[0]),plt.show()

    valid=trainGenerator()
    Y_train = valid.flow_from_dir(YPATH)
    print(Y_train.shape)
    #plt.imshow(Y_train[0]),plt.show()
    #print(X_train.max(), Y_train.min())
    valX = np.array([sift_angle(x) for x in X])
    valY= np.array([sift_angle(y) for y in Y])
    #print(valX.shape, valY.shape)
    #plt.imshow(valX[0]),plt.show()

    train_fit(model, X_train, Y_train, valX, valY)
    #train_generator(model, X_train, Y_train, valX, valY)




if __name__ == '__main__':
    main()
