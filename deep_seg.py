from segmentation.duc_hdc import ResNetDUCHDC
from torchvision import transforms, datasets
from torch.utils import data
from PIL import Image
import glob, os, torch, logging
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt 
from skimage.transform import resize 
import pandas as pd
from utils.metrics import dice_score
TRAIN = True
VAL = False
RUN_ON_TEST = False
TRAIN_NAME = 'train_ids_duc.csv'
VAL_NAME = 'val_ids_duc.csv'
PATH = 'data/train'

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path, batch_size, csv_name,data_augment=False):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = list(pd.read_csv(os.path.join(folder_path,csv_name))['img'])
        self.img_files = [os.path.join(folder_path,'images',os.path.basename(img_path) + '.jpg') for img_path in self.img_files]
        self.mask_files = []
        if not data_augment:
            self.data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
            self.mask_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            ])
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path)[:-4] + '.png') )
        self.batch_size = batch_size
        self.N = len(self.img_files)//batch_size + (len(self.img_files)%batch_size > 0)

    def __getitem__(self, index):
        imgs = []
        masks = []
        for idx in range(self.batch_size*index, self.batch_size*(index+1)):
            img_path = self.img_files[idx]
            mask_path = self.mask_files[idx]
            imgs.append(self.data_transform(Image.open(img_path)))
            masks.append(self.mask_transform(Image.open(mask_path)))
        return torch.stack(imgs), torch.stack(masks).type(torch.LongTensor).view(-1,224,224)

    def __len__(self):
        return self.N

def train(model, train_loader, val_loader, optimizer, epoch, logger, keep_id=None):
    model.train()
    tot_loss = 0
    count = 0
    # criterion = CrossEntropyLoss2d(size_average=False).cuda()
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = data.cuda(), target.cuda()
    #     optimizer.zero_grad()
    #     output = model(data)
    #     if keep_id is not None:
    #         output = output[:, :, keep_id]
    #         target = target[:, keep_id]

    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()
    #     tot_loss += loss.item()
    #     count += data.size()[0]
    #     if batch_idx % 100 == 0:
    #         logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
    #             epoch, batch_idx, len(train_loader),
    #             100. * batch_idx / len(train_loader), loss.item()))
    # tot_loss /= count
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
        train_dataload = DataLoaderSegmentation("data/train",2,TRAIN_NAME)
        val_dataload = DataLoaderSegmentation("data/train",2,VAL_NAME,False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs = 10
        best_val = 0
        for ep in range(epochs):
            _, val = train(model, train_dataload, val_dataload, optimizer, ep+1, logger, keep_id=None)
            if val > best_val :
                torch.save(model.state_dict(),'models/model_duc.pth')
                best_val = val
                logging.info("Model saved at epochs {}".format(ep))

    if VAL:
        model.load_state_dict(torch.load('models/model_duc.pth'))
        model.eval()
        val_dataload = DataLoaderSegmentation("data/train",1,VAL_NAME,False)
        for batch_idx, (data, target) in enumerate(dataload):
            data, target = data, target
            output = np.argmax(np.transpose(model(data).detach().numpy()[0],(1,2,0)),axis=-1)
            goal =  np.transpose(target.detach().numpy(),(1,2,0))
            img = data.detach().numpy()
            img =  np.transpose(img[0],(1,2,0)) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            to_display = np.hstack((np.repeat(output.reshape(224,224,1),3,2),img,np.repeat(goal,3,2)))
            plt.imshow(to_display)
            plt.show()

    if RUN_ON_TEST:
        from utils.loader import test_generator, rle_encode, rle_to_string, rle_decode
        df = pd.read_csv('data/sample_submission.csv')
        model.load_state_dict(torch.load('models/model_duc.pth'))
        model.eval()
        test_gen = test_generator()
        data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        encoded_strings = []
        for img in test_gen:
            pil_img = Image.fromarray(img)
            data = data_transform(pil_img)
            output = resize(np.argmax(np.transpose(model(torch.stack([data])).detach().numpy()[0],(1,2,0)),axis=-1),(720,1280),clip=False,preserve_range=True)
            output[output<0.5] = 0
            output[output>=0.5] = 1
            rle = rle_to_string(rle_encode(output))
            if len(rle) == 0:
                rle = '1 0'
            encoded_strings.append(rle)
        df['rle_mask'] = encoded_strings
        df.to_csv('data/test_submission.csv', index=False)