from segmentation.duc_hdc import ResNetDUCHDC
from segmentation.unet import UNet, AttentionUNet
import cv2
import glob, os, torch, logging
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt 
from skimage.transform import resize 
import pandas as pd
from utils.metrics import dice_score, CrossEntropyLoss2d
from utils.data import DataLoaderSegmentation, train_transform, val_transforms
from utils.data import TRAIN_NAME, VAL_NAME, PATH

import argparse
from typing import Union


MODEL_DICT = {
    "unet": UNet,
    "duc": ResNetDUCHDC,
    "attunet": AttentionUNet
}


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="unet", choices=list(MODEL_DICT.keys()))
parser.add_argument("--model-args", nargs='+', type=int)
parser.add_argument("--lr", '-lr', type=float, default=0.001)
parser.add_argument("--epochs", '-E', default=10, type=int)
parser.add_argument("--batch_size", '-B', default=2, type=int)

args = parser.parse_args()

TRAIN = False
VAL = False
RUN_ON_TEST = False


def train(model, train_loader, val_loader, optimizer, epoch, logger, keep_id=None):
    model.train()
    tot_loss = 0
    count = 0
    criterion = CrossEntropyLoss2d(size_average=False).cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if keep_id is not None:
            output = output[:, :, keep_id]
            target = target[:, keep_id]

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        count += data.size()[0]
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
    tot_loss /= count
    dice_scores = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            output = np.argmax(np.transpose(output.cpu().detach().numpy(),(0,2,3,1)),axis=-1)
            target = target.cpu().detach().numpy()
            for k in range(len(output)):
                dice_scores.append(dice_score(output[k],target[k]))
    logger.info('Val dice score : {}'.format(np.mean(dice_scores)))
    return tot_loss, np.mean(dice_scores)



if __name__=="__main__":

    print(args)
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    import os
    fh = logging.FileHandler(os.path.join("logs", "log.txt"))
    logger.addHandler(fh)
    
    model_class = MODEL_DICT[args.model]
    model_args = args.model_args

    model = model_class(*model_args)
    
    BATCH_SIZE = args.batch_size

    if TRAIN:
        model.cuda()
        train_dataload = DataLoaderSegmentation("data/train", BATCH_SIZE, TRAIN_NAME,
                                                transforms=train_transform)
        val_dataload = DataLoaderSegmentation("data/train", BATCH_SIZE, VAL_NAME,
                                              transforms=val_transforms)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.StepLR(optimizer, 3, 0.9)
        epochs = args.epochs
        best_val = 0
        for ep in range(epochs):
            _, val = train(model, train_dataload, val_dataload, optimizer, ep+1, logger, keep_id=None)
            if val > best_val :
                torch.save(model.state_dict(),'models/model_%s.pth'%args.model)
                best_val = val
                logger.info("Model saved at epochs {}".format(ep))
            scheduler.step()


    if RUN_ON_TEST:
        from utils.loader import test_generator, rle_encode, rle_to_string, rle_decode
        df = pd.read_csv('data/sample_submission.csv')
        model.load_state_dict(torch.load('models/model_%s.pth'%args.model))
        model.eval()
        test_gen = test_generator()
        encoded_strings = []
        for img in test_gen:
            data = val_transforms(image=img)['image']
            output = resize(np.argmax(np.transpose(model(torch.stack([data])).detach().numpy()[0],(1,2,0)),axis=-1),(720,1280),clip=False,preserve_range=True)
            output[output<0.5] = 0
            output[output>=0.5] = 1
            rle = rle_to_string(rle_encode(output))
            if len(rle) == 0:
                rle = '1 0'
            encoded_strings.append(rle)
        df['rle_mask'] = encoded_strings
        df.to_csv('data/test_submission.csv', index=False)
