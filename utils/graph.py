import numpy as np
import copy
import torchvision.models as models
import torch
from torch import nn
import cv2
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_slic_graph(slic, assignments):
    """Recover nodes and neighbors from SLIC assignment.
    
    Parameters
    ----------
    slic
        SLIC model instance (from fast_slic)
    assignments : ndarray
        Cluster index assignments.
    
    Returns
    -------
    nodes : List[Dict]
        List of node info
    neighbors : List[List[int]]
        List of neighbors for each node: `neighbors[i]` is the list of
        indices of the neighbors of node `nodes[i]`
    """
    nodes = copy.deepcopy(slic.slic_model.clusters)
    neighbors = copy.deepcopy(slic.slic_model.get_connectivity(assignments).tolist())
    return nodes, neighbors


def get_node_mask(mask, nodes):
    out = np.empty((len(nodes),), dtype=int)
    for i, n in enumerate(nodes):
        y, x = n['yx']
        y = int(y)
        x = int(x)
        out[i] = mask[y, x, 0] > 0
    return out


def fill_from_assignments(assignments, data):
    """Build an image using the cluster data and assignment matrix."""
    return data[assignments]


def rebuild_mask(img, assignments, labels):
    """Build a mask from the predicted labels and assignment matrix.
    
    Parameters
    ----------
    img : ndarray
        Image for which we want to build a mask
    assignments : ndarray
        Cluster index assignments for each pixel in the image we want to
        get a mask for.
    labels : ndarray
        Labels for each cluster. We assume labels are -1 or 1.
    
    Returns
    -------
    mask : ndarray
        Array of dtype np.uint8 corresponding to a mask image built from the input
        cluster labels vector.
    """
    mask = np.zeros_like(img)  # dtype should be np.uint8
    mask[labels[assignments] > 0] = 255
    return mask


def get_laplacian_node(nodes, neighbors, i: int, data=None, weighted=True):
    """Compute the graph Laplacian at a node.
    
    Parameters
    ----------
    nodes : List[Dict]
        List of graph nodes
    neighbors : List
        List of graph neigbors
    i : int
        Index at which we want the Laplacian
    data : ndarray (optional)
        Data for each graph node we eventually want to use.
        By default the `color` key on the nodes will be used.
    """
    node = nodes[i]
    neigh = neighbors[i]
    if weighted and len(neigh) > 0:
        yx = np.asarray(node['yx'])
        yx_neigh = np.asarray([nodes[j]['yx'] for j in neigh])
        weights = np.linalg.norm(yx - yx_neigh, axis=1, keepdims=True)
    else:
        weights = np.ones((len(neigh),1))
    weights /= weights.sum()
    if data is None:
        col = np.asarray(node['color'])
        cols_neigh = np.asarray([nodes[j]['color'] for j in neigh])
    else:
        col = data[i]
        cols_neigh = data[neigh]
    if len(neigh) > 0:
        lap = np.sum(weights*(cols_neigh-col), axis=0)
    else:
        lap = np.zeros((3,))
    return lap


def get_laplacian(nodes, neighbors, data=None, weighted=True):
    """Get full Laplacian features for each color channel."""
    return np.array([
        get_laplacian_node(nodes, neighbors, i, data, weighted)
        for i in range(len(nodes))
    ])

def get_vgg_features(img):
    """extract global vgg features from input img."""
    dataset_loader = iter(torch.utils.data.DataLoader(np.array([np.transpose(img,axes=[2,0,1]).astype(float)/255]),batch_size=1))
    vgg13 = models.vgg13(pretrained=True)
    vgg13_conv = nn.Sequential(*list(vgg13.children())[:-1]).double()
    for param in vgg13_conv.parameters():
        param.requires_grad = False
    out = vgg13_conv(next(dataset_loader))
    return np.mean(out.numpy()[0],axis=(1,2))
