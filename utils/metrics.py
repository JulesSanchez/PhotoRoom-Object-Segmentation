import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Source: https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
def dice_score(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def score_sample(masks_proposed, mask_truth):
    return np.mean([dice_score(masks_proposed[i],mask_truth[i]) for i in range(len(masks_proposed))])


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

EPSILON = 1e-8


def soft_dice_loss(input: torch.Tensor, labels: torch.Tensor, softmax=True) -> torch.Tensor:
    """Mean soft dice loss over the batch. From Milletari et al. (2016) https://arxiv.org/pdf/1606.04797.pdf 
    
    Parameters
    ----------
    input : Tensor
        (N,C,H,W) Predicted classes for each pixel.
    labels : LongTensor
        (N,C,H,W) Tensor of pixel labels.
    softmax : bool
        Whether to apply `F.softmax` to input to get class probabilities.
    """
    labels = F.one_hot(labels).permute(0, 3, 1, 2)
    dims = (1, 2, 3)  # sum over C, H, W
    if softmax:
        input = F.softmax(input, dim=1)
    intersect = torch.sum(input * labels, dim=dims)
    denominator = torch.sum(input**2 + labels**2, dim=dims)
    ratio = intersect / (denominator + EPSILON)
    return torch.mean(1 - 2. * ratio)


class CombinedLoss(nn.Module):
    """Combined cross-entropy + soft-dice loss."""

    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        self.weight = weight

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CE + Dice"""
        return F.cross_entropy(input, target, reduction="mean", weight=self.weight) \
            + soft_dice_loss(input, target)
