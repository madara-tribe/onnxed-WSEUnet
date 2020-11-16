from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import *
from keras.layers import *
import tensorflow as tf
import cv2
from models.WideResUnet.WideResBlock_UNet import create_model


def create_ensemble_model(model1, model2):
    inputs_ = model1.inputs
    models = [model1, model2]
    outputs = [model.outputs[0] for model in models]
    o = Average()(outputs)
    ensemble_model = Model(inputs_, o, name='ensemble')
    ensemble_model.summary()
    return ensemble_model

#ens_model = create_ensemble_model(fmodel, fmodel2)

#predictions = ensemble_model.predict(test_img)
#print(predictions.shape)
