import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype = torch.float32)
    if op == 'flatten':
        return torch.flatten(x).tolist()
    if op == 'squeeze':
        return torch.squeeze(x).tolist()
    if op == 'transpose':
        return x.T.tolist()
    ### Additional operations: see theory
 
