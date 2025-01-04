import torch
import torch.nn as nn
from torch import tensor


def wasserstein_distance(y_pred: tensor, y_true: tensor) -> tensor:
    """
    Compute the Wasserstein distance between predicted and true time series.

    :param y_pred: Predicted time series tensor
    :param y_true: True time series tensor
    :return: Wasserstein distance scalar
    :raises ValueError: If input tensors don't have 2 or 3 dimensions, shapes don't match, or 3rd dimension isn't 1
    """

    # Handle 3D tensors
    if len(y_pred.shape) == 3:
        if y_pred.size(2) != 1:
            raise ValueError("For 3D tensors, the last dimension must be 1")
        y_pred = y_pred.squeeze(-1)

    if len(y_true.shape) == 3:
        if y_true.size(2) != 1:
            raise ValueError("For 3D tensors, the last dimension must be 1")
        y_true = y_true.squeeze(-1)

    # After potential squeezing, tensors should be 2D
    if len(y_pred.shape) != 2 or len(y_true.shape) != 2:
        raise ValueError(
            "Input tensors must be 2-dimensional (batch_size, sequence_length) or 3-dimensional (batch_size, sequence_length, 1)")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")

    T = y_pred.size(1)  # sequence length

    # Initialize sum for accumulating distances
    total_distance = 0.0

    # Iterate over i from 1 to T
    for i in range(1, T + 1):
        # Calculate cumulative sums up to index i for both sequences
        pred_cumsum = torch.cumsum(y_pred[:, :i], dim=1)
        true_cumsum = torch.cumsum(y_true[:, :i], dim=1)

        # Calculate absolute difference between cumulative sums
        diff = torch.abs(pred_cumsum - true_cumsum)

        # Sum over the j dimension (from 1 to i)
        distance_i = torch.sum(diff, dim=1)

        # Add to total distance
        total_distance += distance_i.mean()

    # Apply final scaling factor
    wasserstein = (2 / (T * (T + 1))) * total_distance

    return wasserstein


class WassersteinLoss(nn.Module):
    """
    Wasserstein distance loss for time series data.

    Implements the Wasserstein distance metric as a PyTorch loss function.
    Supports both 2D tensors (batch_size, sequence_length) and
    3D tensors (batch_size, sequence_length, 1).
    """

    def __init__(self):
        """
        Initialize the WassersteinLoss module.
        """
        super(WassersteinLoss, self).__init__()

    def forward(self, y_pred: tensor, y_true: tensor) -> tensor:
        """
        Compute Wasserstein loss between predicted and true time series.

        :param y_pred: Predicted time series tensor of shape (batch_size, sequence_length) or (batch_size, sequence_length, 1)
        :param y_true: True time series tensor of shape (batch_size, sequence_length) or (batch_size, sequence_length, 1)
        :return: Wasserstein loss scalar
        :raises ValueError: If input tensors don't have valid dimensions or shapes don't match
        
        Example:
             criterion = WassersteinLoss()
             # 2D tensor example
             pred_2d = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
             true_2d = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
             loss_2d = criterion(pred_2d, true_2d)
             print(loss_2d)  # tensor(0.1000)
            
             # 3D tensor example
             pred_3d = torch.tensor([[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]])
             true_3d = torch.tensor([[[0.2], [0.3], [0.4]], [[0.5], [0.6], [0.7]]])
             loss_3d = criterion(pred_3d, true_3d)
             print(loss_3d)  # tensor(0.1000)
        """
        return wasserstein_distance(y_pred, y_true)
