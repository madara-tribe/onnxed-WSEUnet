import os, sys
sys.path.append('../')
os.environ['TF_KERAS'] = '1'
import numpy as np
import keras2onnx
from keras2onnx import convert_keras
import onnxruntime
import onnx
from models.mish_Resnet_RS.Resnet_RS import create_model as create_mishrs_model
from models.SEUNet.WSEUnet import create_model
from models.efficientnet.efficientunet import _get_efficient_unet
from models.efficientnet.efficientnet import get_efficientnet_b5_encoder
from option_parser import get_option

weight_dir = '../weights'
weight_name = 'cp-06.hdf5'
OUTPUT_ONNX_MODEL_NAME = 'unet.onnx'
H=448
W=448
C=3
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
        models.load_weights(os.path.join(weight_dir, weight_path))
    #models.summary()
    return models

def main(config):
    onnx_model_file_name = OUTPUT_ONNX_MODEL_NAME
    models = load_model(config=config, weight_path=weight_name)
    print(models.name)
    
    onnx_model = convert_keras(models, models.name)
    onnx.save(onnx_model, config.model_type+'_'+onnx_model_file_name)
    print("success to output "+config.model_type+'_'+onnx_model_file_name)

if __name__=='__main__':
    cfg = get_option()
    main(cfg)

