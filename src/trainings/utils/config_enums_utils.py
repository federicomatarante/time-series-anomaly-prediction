from torch import nn
from typing import Union, Callable


def get_activation_fn(activation: Union[str, Callable]) -> nn.Module:
    """
    Returns a PyTorch activation function based on the input string or callable.

    :param activation: Name of the activation function as string or a callable that returns nn.Module
                     Available string options: 'relu', 'gelu', 'softmax', 'sigmoid', 'tanh', 'leakyrelu'
    :return: PyTorch activation function module
    :raises ValueError: If the activation string is not recognized
    :raises TypeError: If the activation is neither a string nor callable

    Example:
         # Using string input
         relu = get_activation_fn('relu')
         output = relu(torch.randn(5))

         # Using callable input
         custom_act = get_activation_fn(lambda: nn.ReLU(inplace=True))
         output = custom_act(torch.randn(5))

         # Using different activations
         sigmoid = get_activation_fn('sigmoid')
         tanh = get_activation_fn('tanh')
         leaky = get_activation_fn('leakyrelu')
    """
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "leakyrelu":
            return nn.LeakyReLU()
        raise ValueError(
            f'{activation} is not available. Available activations: relu, gelu, softmax, sigmoid, tanh, leakyrelu')
    elif callable(activation):
        return activation()
    raise TypeError('Activation must be string or callable')
