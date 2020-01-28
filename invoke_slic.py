# change this import if your CPU doesn't support the AVX2 instruction set
from fast_slic.avx2 import SlicAvx2
import numpy as np
from numpy import ndarray
import cv2
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import glob

import argparse

slic = SlicAvx2(num_components=1000, compactness=10)

def build_slic_img(cluster_map: ndarray, clusters):
    out = np.zeros(cluster_map.shape + (3,), dtype=np.uint8)
    for idx in np.ndindex(*cluster_map.shape):
        clus_idx = cluster_map[idx]
        out[idx] = clusters[clus_idx]['color']
    return out



IMAGE_FOLDER = "data/supplementary/test/images"

if __name__ == "__main__":

    img = cv2.imread(glob.glob(IMAGE_FOLDER+"/*.jpg")[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 8))
    plt.subplot(211)
    plt.imshow(img)
    plt.title("Original image")

    cluster_map = slic.iterate(img, max_iter=10)

    segm_ = mark_boundaries(img, cluster_map) * 255
    segm_ = segm_.astype(np.uint8)

    plt.subplot(212)
    plt.imshow(segm_)
    plt.title("Image with superpixel segments")

    plt.tight_layout()





    # cv2.waitKey()

    slic_img = build_slic_img(cluster_map, slic.slic_model.clusters)
    slic_img = cv2.cvtColor(slic_img, cv2.COLOR_Lab2RGB)

    plt.figure()
    plt.imshow(slic_img)
    plt.show()

    # cv2.imshow('image', slic_img, )
    # cv2.waitKey()
    # cv2.destroyAllWindows()


