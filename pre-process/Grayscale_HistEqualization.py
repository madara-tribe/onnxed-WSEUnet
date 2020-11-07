import numpy as np
print(np.__version__) # 1.15.4
import sys
import matplotlib
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage import exposure
import os
import cv2
matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def plot_histogram(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

def display_results(img, img_rescale, img_eq, img_adapteq, cl1):
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 4), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5+i)
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title('Contrast stretching')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title('Histogram equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title('Adaptive equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(cl1, axes[:, 3])
    ax_img.set_title('CLAHE object')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    argvs = sys.argv
    # hh
    hh_image_name = argvs[1]
    hh_image = cv2.imread(hh_image_name, 0)
    plt.imshow(hh_image, 'gray'),plt.show()
    print(np.unique(hh_image))

    # stretching
    p2, p98 = np.percentile(hh_image, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    plot_histogram(img_rescale)
    
    # Equalization
    img_eq = exposure.equalize_hist(hh_image)
    plot_histogram(img_eq)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(hh_image, clip_limit=0.03)
    plot_histogram(img_adapteq)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) => low param show dark, high param show light
    param = 20.0
    clahe = cv2.createCLAHE(clipLimit=param, tileGridSize=(8,8))
    cl1 = clahe.apply(hh_image)
    plot_histogram(cl1)


    #display all hh_image
    display_results(hh_image, img_rescale, img_eq, img_adapteq, cl1)
