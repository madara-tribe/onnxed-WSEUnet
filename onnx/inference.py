import sys, os, cv2
sys.path.append('../')
import onnxruntime
import onnx
import numpy as np
from cal_iou import calc_IoU 
from fbeta_score import eager_binary_fbeta 
H = W = 448
def load_data(config=None):
    imgs, annos = np.load('../data/sar_img.npy'), np.load('../data/sar_anno.npy')
    imgs, annos = imgs.astype(np.float32), annos.astype(np.float32)
    print(imgs.shape, annos.shape)
    return imgs, annos

def onnx_inference(onnx_path):
    imgs, annos = load_data()
    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    pred_annos = session.run(None, {input_name: imgs})[0]
    pred_annos = pred_annos.reshape(len(pred_annos), H, W)
    print('iou score', calc_IoU(annos, pred_annos))
    #print('f1 score', eager_binary_fbeta(annos, pred_annos))

if __name__=='__main__':
    onnx_path = str(sys.argv[1])
    onnx_inference(onnx_path)
