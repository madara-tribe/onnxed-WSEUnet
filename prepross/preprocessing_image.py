# https://note.nkmk.me/python-opencv-numpy-rotate-flip/
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import exposure


# stretching
def stretching(image):
    p2, p98 = np.percentile(image, (2, 98))
    if imamge.shape[2]==3:
        for j in range(3):
            image[:, :, j] = exposure.rescale_intensity(image[:, :, j], in_range=(p2, p98))
        return image
    else:
        image = exposure.rescale_intensity(image, in_range=(p2, p98))
    #plt.imshow(train_stretching[0]),plt.show()
        return image



def clahe(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if image.shape[2]==3:
        for j in range(3):
            image[:, :, j] = clahe.apply(image[:, :, j])
        return image
    else:
        image = clahe.apply(image)
        return image
