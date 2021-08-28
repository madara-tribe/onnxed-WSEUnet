import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.misc import imresize
import sys

# Preprocessing annotation
def process_anno(annos):
    pixel_value = 11
    process_anno=[]
    for idx, img in enumerate(annos):
        # 0, 11, 12 は黒 (0)、それ以外は白(255)
        dst = np.where((img == 0) | (img >= pixel_value), 0, 255)
        #dst = cv2.resize(dst, (256, 256))
        if idx<3:
            plt.imshow(dst, "gray"),plt.show()
        process_anno.append(dst)

    #plt.imshow(process_anno[0], "gray"),plt.show()
    #print(np.unique(process_anno[0]))
    return process_anno



if __name__ == '__main__':
    argvs = sys.argv
    img_path = argvs[1]
    annos, _ = sorted(img_path)
    process_anno = process_anno(annos)
    print(len(process_anno))

    # resize
    process_annos = np.array([cv2.resize(img,(256, 256), interpolation=cv2.INTER_NEAREST) for img in process_anno])
    print(process_annos.shape)
    train_annos, valid_annos = augmentation(process_annos)
    print(train_anno.shape, valid_anno.shape)
    # save
    np.save("pre/train_anno", train_anno)
    np.save("pre/valid_anno", valid_anno)
