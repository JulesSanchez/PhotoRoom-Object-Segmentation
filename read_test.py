import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 

df = pd.read_csv('data/test_submission.csv')
from utils.loader import test_generator, rle_encode, rle_to_string, rle_decode

for index, row in df.iterrows():
    im = rle_decode(row['rle_mask'], (720,1280))
    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(np.asarray(Image.open('data/test/images/' + row['img'] + '.jpg')))
    plt.show()