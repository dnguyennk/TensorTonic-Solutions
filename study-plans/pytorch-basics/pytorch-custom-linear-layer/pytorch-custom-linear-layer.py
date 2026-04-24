import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features)) #match nn.Linear
        self.bias = nn.Parameter(torch.empty(out_features,))
        nn.init.kaiming_uniform_(self.weight, a =math.sqrt(5)) #sqrt(5) Pytorch default, avoid vanish gradients
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0/ math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight.T + self.bias

""" Summary:
1. Kaiming-He uniform (for weight): Designed for ReLU-family activations. The a parameter is the LeakyReLU negative slope.
2. fan_in = number of inputs to a single output neuron = in_features. As fan_in grows, the bound shrinks → bias starts near zero in wide layers. Also an exact copy of nn.Linear's internal behavior
3. Python 3.5+ matrix-multiply operator. Calls __matmul__, which on PyTorch tensors dispatches to torch.matmul (batched, GPU-aware).
4. Every op here (@, .T, +) is a differentiable PyTorch op on tensors with requires_grad=True, autograd builds the computation graph automatically. Calling loss.backward() populates weight.grad and bias.grad with no extra work
"""
