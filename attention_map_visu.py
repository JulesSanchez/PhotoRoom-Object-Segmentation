"""Explainability pipeline."""
import torch
import torch.nn.functional as F

from segmentation import unet
from utils.data import val_transforms
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse

parser = argparse.ArgumentParser(description="Visualisation attention maps from Attenion U-Net.")
parser.add_argument("--img", help="Image file to run the model on.", type=str, required=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model: unet.AttentionUNet = unet.AttentionUNet()
model.load_state_dict(torch.load("models/model_attunet_small.pth"))
model.to(DEVICE)


# Store the visualisations here
VISU_DICT = {}


def attention_vis_hook(module, input: torch.Tensor, output: torch.Tensor):
    # upscale these maps to the original image!
    output = F.interpolate(output, size=(224, 224), mode='bilinear')
    VISU_DICT[module] = output.data.cpu().numpy()


for name, layer in model.named_children():
    if "att" in name:
        print("Registering hook on layer %s" % name)
        layer.register_forward_hook(attention_vis_hook)



args = parser.parse_args()

img = cv2.imread(args.img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# input_res_ = (720, 1280)  # hard coded, actual data resolution
input_res_ = img.shape[2:]
fig = plt.figure(figsize=(12, 5))
plt.subplot(131)
plt.imshow(img)

print("Input resolution:", input_res_)
orig_img = img
augmented_ = val_transforms(image=img)
img = augmented_['image'].to(DEVICE)
img = img.unsqueeze(0)
with torch.no_grad():
    prediction = model(img).cpu()
    probas = F.softmax(prediction, 1)  # actual probability map

plt.subplot(132)
plt.imshow(prediction[0, 1].numpy(), cmap='viridis')
plt.title("Raw network output")

ax = plt.subplot(133)
plt.imshow(probas[0, 1].numpy(), cmap='viridis')
plt.title("Probability map\n(softmax of output)")
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(cax=cax)
plt.tight_layout()


fig: plt.Figure = plt.figure(figsize=(8, 12))

ax_id = 3

ax = fig.add_subplot(3, 2, 1)
ax.imshow(orig_img)
ax.set_title("Initial image")
ax.axis('off')

ax = fig.add_subplot(3, 2, 2)
ims_= ax.imshow(probas[0, 1].numpy(), cmap='viridis')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(ims_, cax=cax)
ax.set_title("Predicted probability map")
ax.axis('off')

for idx, (module, arr) in enumerate(VISU_DICT.items()):
    ax: plt.Axes = fig.add_subplot(3, 2, ax_id)
    ax.imshow(arr[0, 0])
    ax.set_title("Attention map %d" % (idx+1))
    ax_id += 1
    ax.axis('off')
fig.tight_layout()

plt.show()
