import os
import pandas as pd 
from sklearn.model_selection import train_test_split

TRAIN_NAME = 'train_ids_duc.csv'
VAL_NAME = 'val_ids_duc.csv'
VAL_SPLIT = 0.33

list_names = list(pd.read_csv('data/train_ids_clean.csv')['img'])

train_names, val_names = train_test_split(list_names,test_size=VAL_SPLIT)

df_train = pd.DataFrame({'img': train_names})
df_val = pd.DataFrame({'img': val_names})


df_train.to_csv(os.path.join('data/train',TRAIN_NAME), index=False)
df_val.to_csv(os.path.join('data/train',VAL_NAME), index=False)