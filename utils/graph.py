import numpy as np


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
    nodes = slic.slic_model.clusters
    neighbors = slic.slic_model.get_connectivity(assignments).tolist()
    return nodes, neighbors


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
        Labels for each cluster.
    
    Returns
    -------
    mask : ndarray
        Array of dtype np.uint8 corresponding to a mask image built from the input
        cluster labels vector.
    """
    mask = np.zeros_like(img)  # dtype should be np.uint8
    mask[labels[assignments] > 0] = 255
    return mask
