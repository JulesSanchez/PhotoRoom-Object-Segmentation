import numpy as np
from sklearn.model_selection import train_test_split
import pandas 
import os 
import cv2


# Source: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


# Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def get_subset_train(path='data',size=1000):
    list_of_ids = pandas.read_csv(os.path.join(path,'train_ids_clean.csv'),header=0).to_numpy()
    list_of_masks = pandas.read_csv(os.path.join(path,'train_masks_MxuHn2q.csv'),header=0).to_numpy()
    indices = np.random.choice(len(list_of_ids),size=size,replace=False)
    list_of_ids = list_of_ids[indices,0]
    list_of_masks = list_of_masks[indices,1]
    images = [ i for i in range(size)]
    masks = [rle_decode(list_of_masks[i],images[i].shape[:2]) for i in range(size)]
    return images, masks

def split_val_train(path='data',split=0.3,seed=42):
    list_of_ids = pandas.read_csv(os.path.join(path,'train_ids_clean.csv'),header=0).to_numpy()
    return train_test_split(list_of_ids,test_size=split,random_state=seed)


def train_generator(path='data'):
    list_of_ids = pandas.read_csv(os.path.join(path,'train_ids_clean.csv'),header=0).to_numpy()[:,0]
    list_of_masks = pandas.read_csv(os.path.join(path,'train_masks_MxuHn2q.csv'),header=0).to_numpy()[:,1]
    i=0
    while i <= len(list_of_ids):
        img = cv2.imread(os.path.join(os.path.join(path,'train/images'),list_of_ids[i]) + '.jpg')
        yield img, rle_decode(list_of_masks[i],img.shape[:2])
        i+=1


def test_generator(path='data'):
    list_of_ids = pandas.read_csv(os.path.join(path,'test_ids.csv'),header=0).to_numpy()[:,0]
    i=0
    while i < len(list_of_ids):
        yield cv2.imread(os.path.join(os.path.join(path,'test/images'),list_of_ids[i]) + '.jpg')
        i+=1


