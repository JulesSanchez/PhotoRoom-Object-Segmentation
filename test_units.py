"""Unit tests to check if things work (e.g. loaders, model shapes...)."""

from segmentation.unet import UNet

def test_unet():
    net = UNet(3)

    from utils.data import DataLoaderSegmentation, train_transform, TRAIN_NAME

    dataset = DataLoaderSegmentation("data/train", 2, TRAIN_NAME,
                                     transforms=train_transform)

    # tensors, supposedly
    imgs, masks = dataset[5]

    pred_mask = net(imgs)
    print(pred_mask)


if __name__ == "__main__":
    test_unet()
