import math
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class ExistenceOfAnomaly(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold
        # Use float tensors for better performance
        self.add_state("true_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if len(preds.shape) != 3 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 3D tensors of shape [batch_size, channels, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")

        # Vectorized operations across batch
        pred_exists = (preds.sum(dim=(1, 2)) >= self.threshold)
        target_exists = (targets.sum(dim=(1, 2)) >= self.threshold)

        self.true_positives += (pred_exists & target_exists).float().sum()
        self.false_positives += (pred_exists & ~target_exists).float().sum()
        self.false_negatives += (~pred_exists & target_exists).float().sum()

    def compute(self) -> float:
        epsilon = 1e-7  # Prevent division by zero
        numerator = 2 * self.true_positives + epsilon
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives + epsilon
        return (numerator / denominator)

class DensityOfAnomalies(Metric):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("cumulative_density", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if len(preds.shape) != 3 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 3D tensors of shape [batch_size, channels, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")

        # Vectorized absolute difference calculation
        self.cumulative_density += (targets - preds).abs().sum()
        self.total += preds.numel()

    def compute(self) -> float:
        epsilon = 1e-7
        return 1 - (self.cumulative_density / (self.total + epsilon))

class LeadTime(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("cumulative_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_length", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("valid_sequences", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if len(preds.shape) != 3 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 3D tensors of shape [batch_size, channels, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")

        batch_size, channels, window_size = preds.shape
        self.total_length += preds.numel()

        # Create masks for anomalies
        pred_mask = preds >= self.threshold
        target_mask = targets >= self.threshold

        # Find first anomaly indices for all sequences at once
        first_pred = torch.argmax((pred_mask).float(), dim=2)  # [batch_size, channels]
        first_target = torch.argmax((target_mask).float(), dim=2)  # [batch_size, channels]

        # Create validity mask for sequences that have both predicted and actual anomalies
        pred_valid = pred_mask.any(dim=2)  # [batch_size, channels]
        target_valid = target_mask.any(dim=2)  # [batch_size, channels]
        valid_sequences = pred_valid & target_valid

        # Calculate distances only for valid sequences
        distances = torch.abs(first_pred - first_target)
        distances = distances * valid_sequences  # Zero out invalid sequences
        
        self.cumulative_distance += distances.sum()
        self.valid_sequences += valid_sequences.sum()

    def compute(self) -> float:
        epsilon = 1e-7
        return 1 - (self.cumulative_distance / (self.total_length + epsilon))

class DiceScore(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("pred_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("common_predicted_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if len(preds.shape) != 3 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 3D tensors of shape [batch_size, channels, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")

        # Create boolean masks once
        pred_mask = preds >= self.threshold
        target_mask = targets >= self.threshold

        # Use float operations for better performance
        self.pred_positives += pred_mask.float().sum()
        self.target_positives += target_mask.float().sum()
        self.common_predicted_positives += (pred_mask & target_mask).float().sum()

    def compute(self) -> float:
        epsilon = 1e-7
        numerator = 2 * self.common_predicted_positives + epsilon
        denominator = self.pred_positives + self.target_positives + epsilon
        return numerator / denominator
