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


def call_model(weight_path):
    H=256
    W=256
    model_input = Input(shape=(H, W, 3))
    model = create_model(model_input)
    model.load_weights(weight_path)
    return model, model_input
    
# ensemble model
def create_ensemble_model(models):
    model1, model_input = call_model(weight_path1)
    model2, _ = call_model(weight_path2)
    models = [model1, model2]
    outputs = [model.outputs[0] for model in models]
    o = Average()(outputs)

    ensemble_model = Model(model_input, o, name='ensemble')
    ensemble_model.summary()
    return ensemble_model

#predictions = ensemble_model.predict(test_img)
#print(predictions.shape)
