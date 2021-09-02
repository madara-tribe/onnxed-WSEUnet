import numpy as np
import os, cv2
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from models.SEUNet.WSEUnet import create_model
from models.mish_Resnet_RS.Resnet_RS import create_model as create_mishrs_model
from models.efficientnet.efficientunet import _get_efficient_unet
from models.efficientnet.efficientnet import get_efficientnet_b5_encoder
from option_parser import get_option
from cal_iou import calc_IoU
H=int(448)
W=int(448)
C=3

def load_data():
    imgs, annos = np.load('data/sar_img.npy'), np.load('data/sar_anno.npy')
    imgs, annos = imgs.astype(np.float32), annos.astype(np.float32)
    print(imgs.shape, annos.shape)
    return imgs, annos

def load_model(config=None, weight_path=None):
    if config.model_type=='se':
        models = create_model(config.num_cls, input_shape=(H, W, C))
    elif config.model_type=='rs_mish':
        models = create_mishrs_model(config.num_cls, input_shape=(H, W, C))
    elif config.model_type=='efficientunet':
        encoder = get_efficientnet_b5_encoder(input_shape=(H, W, C), pretrained=True)
        models = _get_efficient_unet(encoder, out_channels=config.num_cls,
                          concat_input=True, fpa=None, hypercolumn=None)
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.summary()
    return models

def eval(cfg, weight_path=None):
    X_test, y_test = load_data()
    models = load_model(cfg,weight_path=weight_path)
    pred = models.predict(X_test) 
    #_, acc = models.evaluate(X_test, y_test, verbose=0)
    print(pred.shape, np.unique(pred))
    np.save('pred_test', pred)
    IOU=0
    for im, pr in zip(y_test, pred):
        print(calc_IoU(im, pr.reshape(H, W)))
        IOU += calc_IoU(im, pr.reshape(H, W))
    print('total', IOU/len(y_test))

if __name__=='__main__':
    config=get_option()
    eval(config, weight_path='cp-28.hdf5')


