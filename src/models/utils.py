from torch import nn, Tensor
from torch_geometric.nn import GCNConv


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


def init_mlp_classifier_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):  # Only initialize Linear layers
            if module == list(model.modules())[-1]:  # Last layer
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

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