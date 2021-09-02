import cv2
import PIL
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def remove(im2, im1):
    im_out = np.maximum(im2 - im1, 0.5) - 0.5 
    im_out = np.heaviside(im_out, 0)
    im_out = remove_blob(im_out * 255, threshold_blob_area=25)
    return im_out

def remove_blob(im_in, threshold_blob_area=25): 
    '''remove small blob from your image '''
    im_out = im_in.copy()
    contours, hierarchy = cv2.findContours(im_in.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range (1, len(contours)): 
        index_level = int(hierarchy[0][i][1]) 
        if index_level <= i:
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area <= threshold_blob_area: 
                cv2.drawContours(im_out, [cnt], -1, 0, -1, 1)
    return im_out

def equalizeHist(img):
    for j in range(3):
        img[:, :, j] = cv2.equalizeHist(img[:, :, j])
    return img

class TIFLoad():
    def __init__(self):
        self.resize_w, self.resize_h = 448, 448
        self.input_chanel = 3
        self.num_folder = 28

    def tif_normalize(self, img, StandardScaler=True):
        if StandardScaler:
            ss = preprocessing.StandardScaler()
            if img.ndim==4:
                b, w, h, c = img.shape
                img = img.reshape(b, w*h*c)
                img = ss.fit_transform(img)
                img = img.reshape(b, w, h, c)
            elif img.ndim==3:
                w, h, c = img.shape
                img = img.reshape(c, w*h)
                img = ss.fit_transform(img)
                img = img.reshape(w, h, c)
            elif img.ndim==2:
                img = ss.fit_transform(img)
        else:
            img = img/255
        return img

    def clipping(self, img, clip_max=99.5):
        #img = img_ * 255
        return img.clip(0, clip_max).astype('uint8')

    def create_bgr(self, img1, img2, img3=None):
        r = img1 * 255
        g = img2 * 255
        b = img3 * 255
        return np.dstack((r, g, b))

    def tif_load(self, idx):
        im1 = cv2.imread('./train_images/train_{}/0_VV.tif'.format(idx), -1)
        im2 = cv2.imread('./train_images/train_{}/0_VH.tif'.format(idx), -1)
        im3 = cv2.imread('./train_images/train_{}/1_VV.tif'.format(idx), -1)
        im4 = cv2.imread('./train_images/train_{}/1_VH.tif'.format(idx), -1)
        ims = (im3 - im1)+(im4-im2)
        bgr = self.create_bgr(ims, im1, im2)
        bgr = self.clipping(bgr, clip_max=255)
        bgr = equalizeHist(bgr)
        #c, h, w = im_out.shape
        images = cv2.resize(bgr, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        anno = im3 #cv2.imread('./train_annotations/train_{}.png'.format(idx), -1)
        anno = cv2.resize(anno, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        anno = self.clipping(anno*255, clip_max=255)
        anno = self.tif_normalize(anno, StandardScaler=None)
        images = self.tif_normalize(images, StandardScaler=None)
        print(np.array(anno).shape, np.array(images).shape)
        return np.array(images), np.array(anno)

    def run(self):
        imgs, annos = [], []
        for idx in range(self.num_folder):
            if idx<10:
                idx = '0'+str(idx)
            images, anno = self.tif_load(idx)
            plt.imshow(images),plt.show()
            plt.imshow(anno),plt.show()
            print(anno.shape, images.shape, images.min(), images.max(), anno.min(), anno.max())
            imgs.append(images)
            annos.append(anno)
        imgs = np.array(imgs).reshape(self.num_folder, self.resize_w, self.resize_h, self.input_chanel)
        annos = np.array(annos).reshape(self.num_folder, self.resize_w, self.resize_h)
        print(imgs.shape, annos.shape)
        return imgs, annos
    
if __name__=='__main__':
    imgs, annos = TIFLoad().run()
    print(imgs.max(), imgs.min(), np.unique(annos), imgs.shape, annos.shape)
    np.save('data/sar_img', imgs)
    np.save('data/sar_anno', annos)



