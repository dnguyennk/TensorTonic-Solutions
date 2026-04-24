import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    k = torch.tensor(x, dtype = torch.float32)
    if method == "relu":
        return torch.clamp(k, min = 0).tolist()
    if method == "sigmoid":
        return (1.0 / (1.0 + torch.exp(-k))).tolist()
    if method == "tanh":
        return torch.tanh(k).tolist()
    if method == "leaky_relu":
       return torch.where(k>0, k, 0.01*k).tolist()
        