import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.misc import imresize
from natsort import natsorted
import sys
from data_utils.jupyter_utils import ImageLinePlotter
from data_utils.Augmentation import augmentation, flip_rotate_image, natsort_img

def mask_to_indexmap(masks):
    colormap=[]
    for idx, mask in enumerate(masks):
        mask_h, mask_w, _ = np.shape(mask)
        masked = np.zeros([mask_h, mask_w, 3], dtype=np.uint8)
        for h in range(mask_h):
            for w in range(mask_w):
                class_id = mask[h, w]
                #print(idx, np.unique(class_id))
                r, b, g = (0, 0, 0)
                if class_id == 0:
                    r, g, b = (0, 0, 0) # black
                elif class_id == 11:
                    r, g, b = (0, 0, 0) # black
                elif class_id == 12:
                    r, g, b = (0,128,0) # green
                else:
                    r, g, b = (255,255,255) # white

                masked[h, w, 0] = r
                masked[h, w, 1] = g
                masked[h, w, 2] = b

        plt.imshow(masked),plt.show()
        print(masked.shape)
        colormap.append(masked)
    return colormap
# when save     bgr = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    argvs = sys.argv
    anno_path = argvs[1]
    annos, _ = natsort_img(anno_path)
    print(len(annos), np.unique(annos[0]))

    # resize
    process_annos = np.array([cv2.resize(img,(256, 256), interpolation=cv2.INTER_NEAREST) for img in annos])
    print(process_annos.shape, np.unique(process_annos[0]))
    # colormap
    colormap = mask_to_indexmap(process_annos)
    colormaps = np.vstack([colormap, colormap])
    print(len(colormaps))

    # RGB sample plot
    img_ploter = ImageLinePlotter(0, plot_area_num=4, display_size=5)
    img_ploter.add_image(colormaps[0], title='colormaps A')
    img_ploter.add_image(colormaps[1], title='colormaps B', pos=3)
    img_ploter.add_image(colormaps[2], title='colormaps A', pos=4)
    img_ploter.show_plot()

    # save
    np.save("pre/colormaps_annotation", colormaps)

    # augmentation for train, validation
    train_annos, valid_annos = augmentation(annos)
    np.save("pre/train_annos", train_annos)
    np.save("pre/valid_annos", valid_annos)
