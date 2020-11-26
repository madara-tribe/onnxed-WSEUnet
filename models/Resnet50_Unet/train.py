import numpy as np
import os
import math
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from res50_unet import ResNet_Unet
import numpy as np

model = ResNet_Unet(nClasses=3, input_shape=(128, 128, 3))
sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer=adam, loss = 'mean_squared_error', metrics = ['accuracy'])
model.summary()


class trainGenerator(object):
    def __init__(self):
        self.img_generator=[]
	self.mask_generator=[]
    def clear(self):
	self.img_generator=[]
	self.mask_generator=[]
		
    def flow_from_np(self, X, Y, batch_size=4):
	while:
	    for imgs, mask in zip(X, Y):
		self.img_generator.append(imgs)
		self.mask_generator.append(mask)
		if self.img_generator==batch_size and self.mask_generator==batch_size:
			input_img=np.array(self.img_generator)
			input_mask=np.array(self.mask_generator)
			self.clear()
                        yield input_img, input_mask
					
gene = trainGenerator()
training = gene.flow_from_np(X_train, Y_train, batch_size=4)


class trainGenerator(object):
    def __init__(self):
        self.img_generator=[]

    def flow_from_dir(self, img_path):
        for imgs in os.listdir(img_path):
            if '.DS_Store' in imgs:
                pass
            imgs=cv2.imread(img_path+'/'+imgs)

            if imgs is not None:
                imgs = cv2.resize(imgs, (128, 128), cv2.INTER_NEAREST)
                self.img_generator.append(imgs/ 255)
        input_img=np.array(self.img_generator, dtype=np.float32)
        return input_img
  
train=trainGenerator()
Y_path='/Users/hagiwara/downloads/Datasection/X'
X_train = train.flow_from_dir(Y_path)
print(X_train.shape)
plt.imshow(X_train[0])
plt.show()

valid=trainGenerator()
Y_path='/Users/hagiwara/downloads/Datasection/Y'
Y_train = valid.flow_from_dir(Y_path)
print(Y_train.shape)
plt.imshow(Y_train[0])
plt.show()
print(X_train.max(), Y_train.max())

def flips(X, Y):
    Xs = np.array([img[::-1] for img in X])
    Ys = np.array([img[::-1] for img in Y])
    return Xs, Ys

valX, valY = flips(X_train, Y_train)
print(valX.shape, valY.shape)
plt.imshow(valX[0]),plt.show()
plt.imshow(valY[0]),plt.show()
plt.imshow(X_train[0]),plt.show()
plt.imshow(Y_train[0]),plt.show()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from keras.preprocessing.image import ImageDataGenerator
pixel_level = False
generators = ImageDataGenerator(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

def gene_flow_inputs(datagen, x_image, y_train, batch=2):
    batch = datagen.flow(x_image, y_train, batch_size=batch)
    while True:
        batch_image, batch_mask = batch.next()
        yield batch_image, batch_mask


training = gene_flow_inputs(generators, X_train, Y_train)

model.fit_generator(training, steps_per_epoch=3000, epochs=10,
                         validation_data=(valX, valY))
