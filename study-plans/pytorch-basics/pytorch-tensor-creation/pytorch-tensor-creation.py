import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method == 'zeros':
        return torch.zeros(shape).tolist()
    if method == 'ones':
        return torch.ones(shape).tolist()
    if method == 'full':
        return torch.full(shape, value).tolist()
    if method  == "arange":
        return torch.arange(shape).tolist()
    if method == 'linspace':
        return torch.linspace(shape).tolist()
        