"""Unit tests to check if things work (e.g. loaders, model shapes...)."""

from segmentation.unet import UNet
from segmentation.duc_hdc import ResNetDUCHDCVAE

def test_unet():
    net = UNet(3)

    from utils.data import DataLoaderSegmentation, train_transform, TRAIN_NAME

    dataset = DataLoaderSegmentation("data/train", 2, TRAIN_NAME,
                                     transforms=train_transform)

    # tensors, supposedly
    imgs, masks = dataset[5]

    pred_mask = net(imgs)
    print(pred_mask)

def test_vae():
    net = ResNetDUCHDCVAE(2)

    from utils.data import DataLoaderSegmentation, train_transform, TRAIN_NAME

    dataset = DataLoaderSegmentation("data/train", 2, TRAIN_NAME,
                                     transforms=train_transform)

    # tensors, supposedly
    imgs, masks = dataset[5]
    net.train()
    pred_mask, pred_vae, mean, var = net(imgs)
    print(pred_mask)

    from utils.metrics import KLLoss, L2VAELoss
    criterion1 = KLLoss()
    criterion2 = L2VAELoss()

    loss = criterion1(imgs.size()[1:],mean,var)
    loss2 = criterion2(pred_vae,imgs)


if __name__ == "__main__":
    test_vae()
