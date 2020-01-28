# change this import if your CPU doesn't support the AVX2 instruction set
from fast_slic.avx2 import SlicAvx2
import numpy as np
from numpy import ndarray
import cv2
from skimage.segmentation import mark_boundaries
from skimage import color

import matplotlib.pyplot as plt
import glob

import argparse

slic = SlicAvx2(num_components=400, compactness=10)


IMAGE_FOLDER = "data/test/images"

DEBUG_IMG = True  # whether to debug (show original and segmented image)

if __name__ == "__main__":

    img = cv2.imread(glob.glob(IMAGE_FOLDER+"/*.jpg")[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cluster_map = slic.iterate(img, max_iter=10)

    segm_ = mark_boundaries(img, cluster_map) * 255
    segm_ = segm_.astype(np.uint8)

    if DEBUG_IMG:
        plt.figure(figsize=(6, 8))
        plt.subplot(211)
        plt.imshow(img)
        plt.title("Original image")

        plt.subplot(212)
        plt.imshow(segm_)
        plt.title("Image with superpixel segments")

        plt.tight_layout()


    slic_img = color.label2rgb(cluster_map, img, kind='avg')
    
    # slic_img_segm = mark_boundaries(slic_img, cluster_map) * 255
    # slic_img_segm = slic_img.astype(np.uint8)

    if DEBUG_IMG:    
        plt.figure()
        plt.imshow(slic_img)
        plt.tight_layout()
        plt.show()


