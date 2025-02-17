from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


def any_predicted_anomaly(preds: Tensor, targets: Tensor, threshold: float):
    pred_exists = (preds.sum(dim=2) >= threshold)
    target_exists = (targets.sum(dim=2) >= threshold)

    tp_mask = (pred_exists & target_exists)
    return tp_mask


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
        pred_exists = (preds.sum(dim=(2)) >= self.threshold)
        target_exists = (targets.sum(dim=(2)) >= self.threshold)

        self.true_positives += (pred_exists & target_exists).float().sum(dim=1).mean()
        self.false_positives += (pred_exists & ~target_exists).float().sum(dim=1).mean()
        self.false_negatives += (~pred_exists & target_exists).float().sum(dim=1).mean()

    def compute(self) -> float:
        epsilon = 1e-7  # Prevent division by zero
        numerator = 2 * self.true_positives
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives + epsilon
        return (numerator / denominator)


class DensityOfAnomalies(Metric):
    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("density_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valid_channels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape
        
        # Compute TP mask [B, C]
        pred_sums = preds.sum(dim=2)  # [B, C]
        target_sums = targets.sum(dim=2)
        tp_mask = (pred_sums >= self.threshold) & (target_sums >= self.threshold)
        
        # Compute density per channel (only for TP channels)
        density_per_channel = 1 - (targets.sum(dim=2) - preds.sum(dim=2)).abs() / T  # [B, C]
        density_per_channel = density_per_channel * tp_mask  # Zero out non-TP
        
        # Accumulate
        self.density_sum += density_per_channel.sum()
        self.valid_channels += tp_mask.sum()

    def compute(self):
        return self.density_sum / (self.valid_channels + 1e-7)


class LeadTime(Metric):
    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("lead_time_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valid_channels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape
        
        # TP mask [B, C]
        pred_sums = preds.sum(dim=2)
        target_sums = targets.sum(dim=2)
        tp_mask = (pred_sums >= self.threshold) & (target_sums >= self.threshold)
        
        # Find first anomaly indices
        pred_first = (preds >= self.threshold).float().argmax(dim=2)  # [B, C]
        target_first = (targets >= self.threshold).float().argmax(dim=2)
        
        # Compute lead time only for TP channels
        lead_time = 1 - (pred_first - target_first).abs() / T  # [B, C]
        lead_time = lead_time * tp_mask  # Zero out non-TP
        
        self.lead_time_sum += lead_time.sum()
        self.valid_channels += tp_mask.sum()

    def compute(self):
        return self.lead_time_sum / (self.valid_channels + 1e-7)


class DiceScore(Metric):
    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valid_channels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape
        
        # TP mask [B, C]
        pred_sums = preds.sum(dim=2)
        target_sums = targets.sum(dim=2)
        tp_mask = (pred_sums >= self.threshold) & (target_sums >= self.threshold)
        
        # Compute Dice only for TP channels
        pred_mask = (pred_sums >= self.threshold)  # [B, C, T]
        target_mask = (target_sums >= self.threshold)
        
        intersection = (pred_mask & target_mask).sum(dim=-1)  # [B, C]
        union = pred_mask.sum(dim=-1) + target_mask.sum(dim=-1)
        
        dice = (2 * intersection) / (union + 1e-7)  # [B, C]
        # dice = dice * tp_mask  # Zero out non-TP
        
        self.dice_sum += dice.sum()
        self.valid_channels += tp_mask.sum()

    def compute(self):
        return self.dice_sum / (self.valid_channels + 1e-7)