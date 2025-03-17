from operator import index
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
        self.add_state("total_density_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape

        tp_mask = any_predicted_anomaly(preds, targets, self.threshold)
        sum_mask = tp_mask.float().sum()  # Number of TP channels

        if sum_mask == 0:
            return

        # Compute density per channel (only for TP channels)
        diff = (targets.sum(dim=2) - preds.sum(dim=2)).abs() / T  # [B, C]
        ones = torch.ones_like(diff)

        density_per_channel = (ones - diff) * tp_mask  # Zero out non-TP

        self.density_sum += density_per_channel.sum() / sum_mask
        self.total_density_samples += tp_mask.sum()

    def compute(self):
        return self.density_sum / (self.total_density_samples + 1e-7)


class LeadTime(Metric):
    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("lead_time_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape

        # Time indices tensor (broadcastable across batches/channels)
        # time_indices = torch.arange(T).view(1, 1, T)

        tp_mask = any_predicted_anomaly(preds, targets, self.threshold)
        sum_mask = tp_mask.float().sum()  # Number of TP channels

        if sum_mask == 0:
            return
        # Ground truth: first anomaly index (iy) for each (B, C)
        target_mask = (targets == 1)
        # iy = torch.where(target_mask, time_indices, T).min(dim=2).values  # [B, C]

        # Predictions: first anomaly index (iŷ) for each (B, C)
        pred_mask = (preds >= self.threshold)
        # iy_hat = torch.where(pred_mask, time_indices, T).min(dim=2).values  # [B, C]

        index_pred = torch.zeros(B, C, device=preds.device)
        index_target = torch.zeros(B, C, device=preds.device)
        # Assuming pred_mask is a boolean tensor of shape [B, C, T]
        # Create indices tensor [B, C, T] where each T dimension contains [0, 1, 2, ..., T-1]
        t_indices = torch.arange(T, device=pred_mask.device)[None, None, :].expand(B, C, T)

        # For each [b,c], find the first True index
        # Use float('inf') for positions where no True value exists
        index_pred = torch.where(pred_mask, t_indices, torch.full_like(t_indices, T))
        index_pred = torch.min(index_pred, dim=-1).values

        # Do the same for index_target
        index_target = index_pred  # Since they're using the same condition in the original code

        # Handle cases where no True value exists (optional)
        no_true_mask = ~torch.any(pred_mask, dim=-1)
        index_pred[no_true_mask] = -1  # or any other default value
        index_target[no_true_mask] = -1

        lead_time = torch.ones([B, C], device=preds.device) - ((index_pred - index_target).abs() / T)
        lead_time = lead_time * (tp_mask).float()
        lead_time = lead_time.sum() / tp_mask.float().sum()

        # Accumulate results
        self.lead_time_sum += lead_time.sum()
        self.num_valid += tp_mask.sum()

    def compute(self):
        if self.num_valid == 0:
            return torch.tensor(0.0, device=self.lead_time_sum.device)
        return self.lead_time_sum / self.num_valid


class DiceScore(Metric):
    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valid_channels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape

        # TP mask [B, C]
        tp_mask = any_predicted_anomaly(preds, targets, self.threshold)
        if tp_mask.sum() == 0:
            return

        pred_sums = preds.sum(dim=2)
        target_sums = targets.sum(dim=2)
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
