"""Intepretability module."""
import torch
from captum.attr import (
    LayerActivations
)

from segmentation import unet

model = unet.AttentionUNet()

model.load_state_dict(torch.load("models/model_attunet_small.pth"))
