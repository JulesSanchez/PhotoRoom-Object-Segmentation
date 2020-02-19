from segmentation.duc_hdc import ResNetDUCHDC
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

TRAIN = True
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

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join("logs", "log.txt"))
    logger.addHandler(fh)

    model = ResNetDUCHDC(2)

    if TRAIN:
        model.cuda()
        train_dataload = DataLoaderSegmentation("data/train",2,TRAIN_NAME,
                                                transforms=train_transform)
        val_dataload = DataLoaderSegmentation("data/train",2,VAL_NAME,
                                              transforms=val_transforms)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs = 10
        best_val = 0
        for ep in range(epochs):
            _, val = train(model, train_dataload, val_dataload, optimizer, ep+1, logger, keep_id=None)
            if val > best_val :
                torch.save(model.state_dict(),'models/model_duc.pth')
                best_val = val
                logging.info("Model saved at epochs {}".format(ep))


    if RUN_ON_TEST:
        from utils.loader import test_generator, rle_encode, rle_to_string, rle_decode
        df = pd.read_csv('data/sample_submission.csv')
        model.load_state_dict(torch.load('models/model_duc.pth'))
        model.eval()
        test_gen = test_generator()
        encoded_strings = []
        for img in test_gen:
            data = val_transforms(img)
            output = resize(np.argmax(np.transpose(model(torch.stack([data])).detach().numpy()[0],(1,2,0)),axis=-1),(720,1280),clip=False,preserve_range=True)
            output[output<0.5] = 0
            output[output>=0.5] = 1
            rle = rle_to_string(rle_encode(output))
            if len(rle) == 0:
                rle = '1 0'
            encoded_strings.append(rle)
        df['rle_mask'] = encoded_strings
        df.to_csv('data/test_submission.csv', index=False)