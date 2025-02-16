import torch
import torch.nn as nn
from torch import Tensor


def wasserstein_distance(y_pred: Tensor, y_true: Tensor, apply_scaling_factor=False, weight: int = None) -> Tensor:
    """
    Compute the Wasserstein distance between predicted and true time series.
    :param y_pred: Predicted time series tensor. Shape: (batch_size, channels, window_size)
    :param y_true: True time series tensor. Shape: (batch_size, channels, window_size)
    :param apply_scaling_factor: whether to apply scaling factor or not.
    :param weight: weight to apply to samples in batches which have at least one anomaly. Samples without anomalies are
        considered with 1 weight. If not set, the loss is not weighted.
    :return: Wasserstein distance scalar
    :raises ValueError: If input tensors don't have 3 dimensions, shapes don't match
    """

    if len(y_pred.shape) != 3 or len(y_true.shape) != 3:
        raise ValueError(
            "Input tensors must be 3-dimensional (batch_size, channels, window_size)")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")

    C = y_pred.size(1)  # Number of channels
    abs_diff = torch.abs(y_pred - y_true)  # Shape: [B, C, T]
    T = abs_diff.shape[2]  # Seq length
    mask = torch.triu(torch.ones(T, T)).to(abs_diff.device)  # Shape: [T, T]
    abs_diff = abs_diff.transpose(1, 2)  # Shape: [B, T, C]
    masked_sums = abs_diff.unsqueeze(2) * mask.unsqueeze(0).unsqueeze(-1)

    channel_sums = masked_sums.sum(dim=2)  # Shape: [B, T, C]
    channel_sums = channel_sums.sum(dim=1)  # Shape: [B, C]
    scaling = (2 / (T * C * (T + 1))) if apply_scaling_factor else 1
    wass_2 = scaling * channel_sums.sum(dim=1)  # Scalar
    if weight:
        condition = (y_true.flatten(1, 2) > 0).any(dim=-1)
        inv_condition = ~condition
        weights: Tensor = condition * weight + inv_condition * 1
        weighted_wass = torch.sum(wass_2 * weights) / torch.sum(weights)
        return weighted_wass
    else:
        return wass_2.mean()


class WassersteinLoss(nn.Module):
    """
    Wasserstein distance loss for time series data.

    Implements the Wasserstein distance metric as a PyTorch loss function.
    Supports both 2D tensors (batch_size, sequence_length) and
    3D tensors (batch_size, sequence_length, 1).

    :param apply_scaling_factor: whether to apply scaling factor or not.
    :param weight: weight to apply to samples in batches which have at least one anomaly. Samples without anomalies are
        considered with 1 weight. If not set, the loss is not weighted.
    """

    def __init__(self, apply_scaling_factor=True, weight=None):
        """
        Initialize the WassersteinLoss module.
        """
        super(WassersteinLoss, self).__init__()
        self.apply_scaling_factor = apply_scaling_factor
        self.weight = weight

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
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
        return wasserstein_distance(y_pred, y_true, apply_scaling_factor=self.apply_scaling_factor, weight=self.weight)


class HuberLoss(nn.Module):
    """
    Huber Loss module for PyTorch neural networks.
    Combines quadratic and linear loss, making it robust to outliers
    while maintaining MSE-like behavior for small errors.

    :param delta: Threshold for switching between L1 and L2 loss
    :param reduction: Specifies reduction method - 'none' for no reduction,
                     'mean' for mean of losses, 'sum' for sum of losses
    :raises ValueError: If reduction is not one of ['none', 'mean', 'sum'] or if delta <= 0


    :return: Loss value - scalar if reduction='mean'/'sum',
             or shape (batch_size, *) if reduction='none'
    :rtype: torch.Tensor

    The loss function transitions smoothly between two regimes:
    - Quadratic (L2) loss for errors smaller than delta
    - Linear (L1) loss for errors larger than delta
    """

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction: {reduction}")
        if delta <= 0:
            raise ValueError(f"Delta must be positive, got {delta}")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        :param y_pred: Predicted values from the model. Tensor of shape [Batch Size, Channels, Timesteps]
        :param y_true: Ground truth values. Tensor of shape [Batch Size, Channels, Timesteps]
        :raises ValueError: If shapes of y_pred and y_true don't match.
        :return: tensor loss. Tensor of shape [1]
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}")

        # Calculate error
        error = y_true - y_pred
        abs_error = torch.abs(error)

        # quadratic part (MSE-like)
        quadratic = 0.5 * torch.square(error)
        # linear part (MAE-like)
        linear = self.delta * abs_error - 0.5 * self.delta ** 2

        loss = torch.where(abs_error <= self.delta, quadratic, linear)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss
