"""Unit tests to check if things work (e.g. loaders, model shapes...)."""

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


if __name__ == "__main__":
    test_unet()
    # test_attunet()
