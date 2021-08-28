import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

class TIFLoad():
    def __init__(self):
        self.H, self.W = 1792, 1792
        self.stepSize = 224
        self.w_width, self.w_height = 448, 448
        self.w_batch = 36
        self.input_chanel = 4
        self.num_folder = 28
        
    def sliding_window(self, image_):
        Windows = []
        image = cv2.resize(image_, (self.H, self.W), interpolation=cv2.INTER_NEAREST)
        for x in range(0, image.shape[1] - self.w_width , self.stepSize):
            for y in range(0, image.shape[0] - self.w_height, self.stepSize):
                window = image[x:x + self.w_width, y:y + self.w_height]
                if window.shape[0]==self.w_width and window.shape[1]==self.w_height:
                    Windows.append(window)
        return np.array(Windows).reshape(1, self.w_batch, self.w_width, self.w_height)


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

    def clipping(self, img_, clip_max=99.5):
        img = img_ * 255
        return img.clip(0, clip_max).astype('uint8')

    def tif_load(self, idx, sliding_w=True):
        if sliding_w:
            im1 = self.sliding_window(cv2.imread('./train_images/train_{}/0_VV.tif'.format(idx), -1))
            im2 = self.sliding_window(cv2.imread('./train_images/train_{}/0_VH.tif'.format(idx), -1))
            im3 = self.sliding_window(cv2.imread('./train_images/train_{}/1_VV.tif'.format(idx), -1))
            im4 = self.sliding_window(cv2.imread('./train_images/train_{}/1_VH.tif'.format(idx), -1))
            images = np.vstack([im1, im2, im3, im4])
            images = images.reshape(self.w_batch, self.w_width, self.w_height, self.input_chanel)

            anno = self.sliding_window(cv2.imread('./train_annotations/train_{}.png'.format(idx), -1))
        else:
            images = np.array([cv2.imread('./train_images/train_{}/0_VV.tif'.format(idx), -1),
                  cv2.imread('./train_images/train_{}/0_VH.tif'.format(idx), -1), 
                  cv2.imread('./train_images/train_{}/1_VV.tif'.format(idx), -1), 
                  cv2.imread('./train_images/train_{}/1_VH.tif'.format(idx), -1)])
            c, h, w = images.shape
            images = cv2.resize(images.reshape(h, w, self.input_chanel), (self.w_width, self.w_height), interpolation=cv2.INTER_NEAREST)

            anno = cv2.imread('./train_annotations/train_{}.png'.format(idx), -1)
            anno = cv2.resize(anno, (self.w_width, self.w_height), interpolation=cv2.INTER_NEAREST)
        images = self.clipping(images, clip_max=99.5)
        print(np.array(anno).shape, np.array(images).shape)
        return np.array(images), np.array(anno)

    def run(self, sliding_w = True):
        imgs, annos = [], []
        for idx in range(self.num_folder):
            if idx<10:
                idx = '0'+str(idx)        
            images, anno = self.tif_load(idx, sliding_w=sliding_w)
            print(anno.shape, images.shape, images.min(), images.max(), anno.min(), anno.max())
            imgs.append(images)
            annos.append(anno)
        if sliding_w:
            imgs = np.array(imgs).reshape((self.num_folder*self.w_batch), self.w_width, self.w_height, self.input_chanel)
            annos = np.array(annos).reshape((self.num_folder*self.w_batch), self.w_width, self.w_height)
        else:
            imgs = np.array(imgs).reshape(self.num_folder, self.w_width, self.w_height, self.input_chanel)
            annos = np.array(annos).reshape(self.num_folder, self.w_width, self.w_height)
        print(imgs.shape, annos.shape)
        imgs = self.tif_normalize(imgs, StandardScaler=True)
        return imgs, annos

    
if __name__=='__main__':
    imgs, annos = TIFLoad().run(sliding_w = True)
    print(imgs.max(), imgs.min(), np.unique(annos))
    #plt.imshow(imgs[3].reshape(4, 448, 448)[0], 'gray'),plt.show()
    #plt.imshow(annos[3], 'gray'),plt.show()
    np.save('data/sar_img', imgs)
    np.save('data/sar_anno', annos)



