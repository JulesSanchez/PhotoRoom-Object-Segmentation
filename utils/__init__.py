

def get_slic_graph(slic, assignments):
    """Recover nodes and neighbors from SLIC assignment."""
    nodes = slic.slic_model.clusters
    neighbors = slic.slic_model.get_connectivity(assignments).tolist()
    return nodes, neighbors
