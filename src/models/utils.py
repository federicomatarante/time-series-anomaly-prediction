from torch import nn, Tensor
from torch_geometric.nn import GCNConv

from src.trainings.utils.config_enums_utils import get_activation_fn


class Permute(nn.Module):
    """
    Torch Module for input permutation.
    :param dims: dimensions to permute.
    """

    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)


class Flatten(nn.Module):
    """
    Torch Module for flatten permutation.
    :param dims: dimensions to flatten.
    """

    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(*self.dims)


class UnFlatten(nn.Module):
    """
    Torch Module for flatten permutation.
    """

    def __init__(self, dim: int, *unflattened_size: int):
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, x: Tensor) -> Tensor:
        return x.unflatten(self.dim, self.unflattened_size)


class ActivationFunction(nn.Module):
    """
    Torch module for activation function
    """

    def __init__(self, act_name: str):
        super().__init__()
        self.act = get_activation_fn(act_name)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


def init_transformer_encoder_weights(model):
    for module in list(model.modules()):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.xavier_uniform_(module.weight)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    return model


def init_mlp_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):  # Only initialize Linear layers
            nn.init.kaiming_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return model


def init_gcn_weights(model):
    for module in list(model.modules()):
        if isinstance(module, nn.Linear):
            # Linear layers - using Kaiming for ReLU-based networks
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, GCNConv):
            # GCNConv layers have a weight matrix for feature transformation
            nn.init.kaiming_uniform_(module.lin.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return model
