import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt

def augmentation(annos):
    annos, flip, flip_up, flip_up_lr, right_90rotate, left_90rotate = flip_rotate_image(annos)
    #plt.imshow(annos[0], "gray"),plt.show()
    train_annos=np.vstack([annos, flip, flip_up, flip_up_lr, right_90rotate])
    valid_annos=left_90rotate
    print(train_annos.shape, valid_annos.shape)
    return train_annos, valid_annos

def fliplr(X, Y):
    Xs = np.array([np.fliplr(img) for img in X])
    Ys = np.array([np.fliplr(img) for img in Y])
    return Xs, Ys


def sift_angle(image, y_move_ratio=0, x_move_ratio=0, angle_ratio=float(np.pi/60)):
    h, w, _ = np.shape(image)
    size = tuple(np.array([w, h]))
    print(size)
    #np.pi=3.141592653589793
    rad=angle_ratio
    move_x = x_move_ratio
    move_y = w * y_move_ratio

    matrix = [[np.cos(rad), -1 * np.sin(rad), move_x],
                   [np.sin(rad), np.cos(rad), move_y]]

    affine_matrix = np.float32(matrix)
    chage_angle = cv2.warpAffine(image, affine_matrix, size, flags=cv2.INTER_LINEAR)
    return chage_angle
    
def flip_rotate_image(image):
    # flip
    flip =np.array([cv2.flip(img, 1) for img in image])
    # flip_up
    flip_up =np.array([cv2.flip(img, 0) for img in image])
    #print(flip_up.shape, np.unique(flip_up[0]))
    #plt.imshow(flip_up[0], "gray"),plt.show()

    # flip_up_lr
    flip_up_lr =np.array([cv2.flip(img, -1) for img in image])

    # right_90rotate
    right_90rotate = np.array([cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in image])

    # left_90rotate
    left_90rotate = np.array([cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in image])

    return image, flip, flip_up, flip_up_lr, right_90rotate, left_90rotate
