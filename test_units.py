"""Unit tests to check if things work (e.g. loaders, model shapes...)."""

from segmentation.unet import UNet, AttentionUNet

def test_unet():
    net = UNet(3)

    from utils.data import DataLoaderSegmentation, train_transform, TRAIN_NAME

    dataset = DataLoaderSegmentation("data/train", 2, TRAIN_NAME,
                                     transforms=train_transform)

    # tensors, supposedly
    imgs, masks = dataset[5]
    img = imgs[0]
    img = img.view(-1, *img.shape)
    print("input shape:", img.shape)

    pred_mask = net(img)
    print(pred_mask)


def test_attunet():
    net = AttentionUNet(3)

    from utils.data import DataLoaderSegmentation, train_transform, TRAIN_NAME

    dataset = DataLoaderSegmentation("data/train", 2, TRAIN_NAME,
                                     transforms=train_transform)

    # tensors, supposedly
    imgs, masks = dataset[5]
    img = imgs[0]
    print("img shape", img.shape)
    pred_mask = net(img)
    print(pred_mask)


if __name__ == "__main__":
    test_unet()
