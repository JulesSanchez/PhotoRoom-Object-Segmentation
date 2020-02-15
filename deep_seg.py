from segmentation.duc_hdc import ResNetDUCHDC
from torchvision import transforms, datasets
from torch.utils import data
from PIL import Image
import glob, os, torch, logging
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path, batch_size):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.jpg'))
        self.mask_files = []
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

def train(model, train_loader, optimizer, epoch, logger, keep_id=None):
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
        if batch_idx % 5 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
    tot_loss /= count
    return tot_loss

if __name__=="__main__":

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join("logs", "log.txt"))
    logger.addHandler(fh)

    dataload = DataLoaderSegmentation("data/train",2)

    model = ResNetDUCHDC(2)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, dataload, optimizer, 10, logger, keep_id=None)