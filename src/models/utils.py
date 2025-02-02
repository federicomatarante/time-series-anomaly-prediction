from torch import nn, Tensor


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

