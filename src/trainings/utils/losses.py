import torch
import torch.nn as nn
from torch import tensor


def wasserstein_distance(y_pred: tensor, y_true: tensor, apply_scaling_factor=False) -> tensor:
    """
    Compute the Wasserstein distance between predicted and true time series.
    :param y_pred: Predicted time series tensor. Shape: (batch_size, channels, window_size)
    :param y_true: True time series tensor. Shape: (batch_size, channels, window_size)
    :return: Wasserstein distance scalar
    :raises ValueError: If input tensors don't have 3 dimensions, shapes don't match
    """

    if len(y_pred.shape) != 3 or len(y_true.shape) != 3:
        raise ValueError(
            "Input tensors must be 3-dimensional (batch_size, channels, window_size)")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")

    T = y_pred.size(2)  # sequence length
    C = y_pred.size(1)
    # Initialize sum for accumulating distances
    total_distance = 0.0
    # Iterate over i from 1 to T
    for i in range(1, T + 1):
        # Calculate cumulative sums up to index i for both sequences for each channel
        pred_cumsum = torch.cumsum(y_pred[:, :, :i], dim=2)
        true_cumsum = torch.cumsum(y_true[:, :, :i], dim=2)

        # Calculate absolute difference between cumulative sums for each channel
        diff = torch.abs(pred_cumsum - true_cumsum)

        # Sum over the dimension of the channels and the window (from 1 to i)
        distance_i = torch.sum(diff, dim=(1, 2))

        # Add to total distance
        total_distance += distance_i.mean()

    # Apply final scaling factor
    scaling = (2 / (T * C * (T * C + 1))) if apply_scaling_factor else 1
    wasserstein = scaling * total_distance
    return wasserstein


class WassersteinLoss(nn.Module):
    """
    Wasserstein distance loss for time series data.

    Implements the Wasserstein distance metric as a PyTorch loss function.
    Supports both 2D tensors (batch_size, sequence_length) and
    3D tensors (batch_size, sequence_length, 1).
    """

    def __init__(self, apply_scaling_factor=True):
        """
        Initialize the WassersteinLoss module.
        """
        super(WassersteinLoss, self).__init__()
        self.apply_scaling_factor = apply_scaling_factor

    def forward(self, y_pred: tensor, y_true: tensor) -> tensor:
        """
        Compute Wasserstein loss between predicted and true multivariate time series.
        :param y_pred: Predicted time series tensor of shape (batch_size, channels, sequence_length)
        :param y_true: True time series tensor of shape (batch_size, channels. sequence_length)
        :return: Wasserstein loss scalar.
        :raises ValueError: If input tensors don't have valid dimensions or shapes don't match
        
         Example:
             criterion = WassersteinLoss()

             # 2D tensor example (batch_size=2, sequence_length=5)
             pred_2d = torch.tensor([
                 [1.2, 2.3, 2.1, 1.8, 2.5],  # First sequence
                 [3.1, 3.3, 3.0, 3.4, 3.2]   # Second sequence
             ])
             true_2d = torch.tensor([
                 [1.0, 2.0, 2.0, 2.0, 2.2],  # Ground truth for first sequence
                 [3.0, 3.5, 3.2, 3.3, 3.0]   # Ground truth for second sequence
             ])
             loss_2d = criterion(pred_2d, true_2d)
             print(loss_2d)  # Example output: tensor(0.2345)

             # 3D tensor example (batch_size=2, channels=2, sequence_length=4)
             pred_3d = torch.tensor([
                 [  # First batch item
                     [1.1, 1.3, 1.2, 1.4],  # Channel 1
                     [2.1, 2.4, 2.3, 2.2]   # Channel 2
                 ],
                 [  # Second batch item
                     [1.2, 1.4, 1.3, 1.5],  # Channel 1
                     [2.2, 2.5, 2.4, 2.3]   # Channel 2
                 ]
             ])
             true_3d = torch.tensor([
                 [  # First batch item
                     [1.0, 1.2, 1.2, 1.3],  # Channel 1
                     [2.0, 2.3, 2.2, 2.1]   # Channel 2
                 ],
                 [  # Second batch item
                     [1.1, 1.3, 1.2, 1.4],  # Channel 1
                     [2.1, 2.4, 2.3, 2.2]   # Channel 2
                 ]
             ])
             loss_3d = criterion(pred_3d, true_3d)
             print(loss_3d)  # Example output: tensor(0.1123)
        """
        return wasserstein_distance(y_pred, y_true, apply_scaling_factor=self.apply_scaling_factor)
