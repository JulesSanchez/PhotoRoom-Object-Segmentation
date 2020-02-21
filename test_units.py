"""Unit tests to check if things work (e.g. loaders, model shapes...)."""

from segmentation.duc_hdc import ResNetDUCHDCVAE
from segmentation.unet import UNet, AttentionUNet
from utils.data import DataLoaderSegmentation, train_transform, TRAIN_NAME

def test_unet():
    net = UNet(3)
    import ipdb; ipdb.set_trace()

    dataset = DataLoaderSegmentation("data/train", 2, TRAIN_NAME,
                                     transforms=train_transform)

    # tensors, supposedly
    imgs, masks = dataset[5]
    img = imgs[0]
    img = img.view(-1, *img.shape)
    print("input shape:", img.shape)

    pred_mask = net(img)
    print(pred_mask)
    print(pred_mask.shape)


def test_attunet():
    net = AttentionUNet(3)

    dataset = DataLoaderSegmentation("data/train", 2, TRAIN_NAME,
                                     transforms=train_transform)

    # tensors, supposedly
    imgs, masks = dataset[5]
    img = imgs[0]
    img = img.view(-1, *img.shape)
    print("img shape", img.shape)
    pred_mask = net(img)
    print(pred_mask)
    print(pred_mask.shape)

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
    test_unet()
    # test_attunet()
