import os
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
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
