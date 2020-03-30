# PhotoRoom-Object-Segmentation
Repository of the data challenge for the "Modèles Multiéchelles et Réseaux de Neurones Convolutifs" course

Pour DUC HDC :

creer un folder logs
creer un folder models
dans models : mettre resnet152-b121ed2d.pth (resnet152 pytorch)

## Training

```
python deep_seg.py
```
Use
```bash
python deep_seg.py -h
```

Visualize the training curves and prediction images:
```bash
tensorboard --logdir runs/
```

## Visualize attention maps

To visualize the attention maps of the Attention U-Net model, see script [attention_map_visu.py](attention_map_visu.py)
```bash
python attention_map_visu.py --img path/to/image
```
![attention maps from 1st training image](attention_maps_0000_0000_train.png)

## Train-test split

```bash
python utils/create_val.py
```
